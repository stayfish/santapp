"""Notebook-compatible SANTA++ attention for Hugging Face Youtu models."""

from __future__ import annotations

import math
import types
from typing import Any

import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from transformers import YoutuPreTrainedModel
from transformers.cache_utils import Cache
from transformers.models.youtu.modeling_youtu import YoutuAttention

from .clustering import ClusterSummary, cluster_old_prefix


# Runtime state follows the notebook's per-layer cache lifecycle. Clustering
# itself lives in santapp.clustering; attention stores only returned summaries.
_SANTAPP_SUMMARIES: dict[tuple[Any, ...], tuple[int, dict[tuple[int, int], ClusterSummary]]] = {}
_SANTAPP_BOUNDARIES: dict[int, int] = {}
_SANTAPP_PREFILL_QUERIES: dict[int, torch.Tensor] = {}


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate ``x=[...,D]`` by swapping and negating its two half vectors."""
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Apply non-interleaved RoPE to ``query,key=[B,H,T,D]``.

    The rotation is :math:`x'=x\cos\theta+R(x)\sin\theta`, where
    :math:`R` swaps the two half vectors and negates the former second half.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


def apply_rotary_pos_emb_interleave(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the notebook's interleaved RoPE to ``query,key=[B,H,T,D]``."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    def pair_to_halves(x: torch.Tensor) -> torch.Tensor:
        batch, heads, tokens, width = x.shape
        return x.view(batch, heads, tokens, width // 2, 2).transpose(4, 3).reshape(
            batch, heads, tokens, width
        )

    query = pair_to_halves(query)
    key = pair_to_halves(key)
    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


def santapp_attn(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    mode: str,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    r"""Run notebook SANTA++ attention for a one-token decode.

    Inputs have shapes ``query_states=[B,H,1,D]``,
    ``key_states=[B,H,K,D]`` and ``value_states=[B,H,K,Dv]``. ``topk`` ranks
    clusters by :math:`\log|G|+\alpha q^T\bar{k}_G` and reads complete
    clusters until ``sample_count`` tokens are covered. The recent window is
    always evaluated exactly.
    """
    del attention_mask, args
    if query_states.ndim != 4 or query_states.shape[-2] != 1:
        raise ValueError("santapp_attn expects query_states=[B,H,1,D]")
    if key_states.ndim != 4 or value_states.ndim != 4:
        raise ValueError("key_states and value_states must be [B,H,K,D]")

    mode = {"santapp": "guided"}.get(mode, mode)
    if mode not in {"topk", "guided", "santa", "uniform"}:
        raise ValueError(f"Unknown SANTA++ mode: {mode!r}")
    sample_count = int(kwargs.get("sample_count", 128))
    recent_window = int(kwargs.get("recent_window", 64))
    group_size = int(kwargs.get("group_size", 16))
    feature_mode = kwargs.get("cluster_feature", "fingerprint")
    probe_count = int(kwargs.get("probe_count", 64))
    probe_start_fraction = float(kwargs.get("probe_start_fraction", 0.75))
    cluster_seed = int(kwargs.get("cluster_seed", 0))
    kmeans_iterations = int(kwargs.get("kmeans_iterations", 100))
    summary_key = int(kwargs.get("summary_key", 0))

    total_tokens = key_states.shape[-2]
    old_end = _SANTAPP_BOUNDARIES.setdefault(
        summary_key, max(0, total_tokens - recent_window)
    )
    if old_end == 0:
        raise ValueError("Sparse attention requires a non-empty old prefix")

    query = query_states[..., 0, :].float()
    recent_scores = (
        key_states[..., old_end:, :].float() * query.unsqueeze(-2)
    ).sum(dim=-1) * scaling
    recent_values = value_states[..., old_end:, :].float()
    batch, heads = query.shape[:2]
    outputs = torch.empty(
        batch,
        heads,
        value_states.shape[-1],
        device=query.device,
        dtype=torch.float32,
    )

    summaries = None
    if mode in {"topk", "guided"}:
        cache_key = (
            summary_key,
            feature_mode,
            group_size,
            probe_count,
            probe_start_fraction,
            cluster_seed,
        )
        cached = _SANTAPP_SUMMARIES.get(cache_key)
        if cached is None:
            if summary_key not in _SANTAPP_PREFILL_QUERIES:
                raise RuntimeError("Run a dense prefill before clustered decode")
            summaries = cluster_old_prefix(
                key_states,
                _SANTAPP_PREFILL_QUERIES[summary_key],
                old_end,
                feature_mode=feature_mode,
                group_size=group_size,
                scaling=scaling,
                probe_count=probe_count,
                probe_start_fraction=probe_start_fraction,
                seed=cluster_seed,
                kmeans_kwargs={"max_iter": kmeans_iterations},
            )
            _SANTAPP_SUMMARIES[cache_key] = (old_end, summaries)
        else:
            old_end, summaries = cached
            recent_scores = (
                key_states[..., old_end:, :].float() * query.unsqueeze(-2)
            ).sum(dim=-1) * scaling
            recent_values = value_states[..., old_end:, :].float()

    for batch_id in range(batch):
        for head_id in range(heads):
            summary = summaries[(batch_id, head_id)] if summaries is not None else None
            if summary is not None:
                group_logits = (
                    summary.key_centroids @ query[batch_id, head_id] * scaling
                    + summary.lengths_float.log()
                )

            if mode == "topk":
                selected: list[torch.Tensor] = []
                selected_count = 0
                for group_id in group_logits.argsort(descending=True).tolist():
                    start = int(summary.starts[group_id].item())
                    length = int(summary.lengths_long[group_id].item())
                    selected.append(summary.members[start : start + length])
                    selected_count += length
                    if selected_count >= sample_count:
                        break
                indices = torch.cat(selected)
                selected_scores = (
                    key_states[batch_id, head_id, indices].float()
                    @ query[batch_id, head_id]
                    * scaling
                )
                exact_scores = torch.cat(
                    (selected_scores, recent_scores[batch_id, head_id])
                )
                exact_values = torch.cat(
                    (
                        value_states[batch_id, head_id, indices].float(),
                        recent_values[batch_id, head_id],
                    )
                )
                outputs[batch_id, head_id] = (
                    torch.softmax(exact_scores, dim=0) @ exact_values
                )
                continue

            if mode == "guided":
                group_probability = torch.softmax(group_logits, dim=0)
                sampled_groups = torch.multinomial(
                    group_probability, sample_count, replacement=True
                )
                lengths = summary.lengths_long[sampled_groups]
                offsets = (
                    torch.rand(sample_count, device=query.device) * lengths.float()
                ).long()
                positions = summary.starts[sampled_groups] + offsets
                indices = summary.members[positions]
                log_proposal = (
                    group_probability[sampled_groups].log()
                    - summary.lengths_float[sampled_groups].log()
                )
            elif mode == "santa":
                old_scores = (
                    key_states[batch_id, head_id, :old_end].float()
                    @ query[batch_id, head_id]
                    * scaling
                )
                log_probability = old_scores - torch.logsumexp(old_scores, dim=0)
                indices = torch.multinomial(
                    log_probability.exp(), sample_count, replacement=True
                )
                log_proposal = log_probability[indices]
            else:
                indices = torch.randint(
                    old_end, (sample_count,), device=query.device
                )
                log_proposal = torch.full(
                    (sample_count,), -math.log(old_end), device=query.device
                )

            sampled_scores = (
                key_states[batch_id, head_id, indices].float()
                @ query[batch_id, head_id]
                * scaling
            )
            sampled_log_weights = sampled_scores - log_proposal
            maximum = torch.cat(
                (sampled_log_weights, recent_scores[batch_id, head_id])
            ).max()
            sampled_weights = (sampled_log_weights - maximum).exp() / sample_count
            recent_weights = (recent_scores[batch_id, head_id] - maximum).exp()
            numerator = (
                sampled_weights @ value_states[batch_id, head_id, indices].float()
            )
            numerator += recent_weights @ recent_values[batch_id, head_id]
            outputs[batch_id, head_id] = numerator / (
                sampled_weights.sum() + recent_weights.sum()
            )

    return outputs.unsqueeze(-2).to(query_states.dtype), None


def _youtu_attention_forward(
    self: YoutuAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run notebook Youtu MLA for ``hidden_states=[B,T,hidden_size]``."""
    if hidden_states.ndim != 3:
        raise ValueError("Expected hidden_states=[B,T,hidden_size]")
    batch_size, sequence_length = hidden_states.shape[:-1]
    if sequence_length > 1:
        _SANTAPP_BOUNDARIES.pop(self.layer_idx, None)
        _SANTAPP_PREFILL_QUERIES.pop(self.layer_idx, None)
        for key in [key for key in _SANTAPP_SUMMARIES if key[0] == self.layer_idx]:
            _SANTAPP_SUMMARIES.pop(key)

    query_shape = (batch_size, sequence_length, -1, self.qk_head_dim)
    key_shape = (
        batch_size,
        sequence_length,
        -1,
        self.qk_nope_head_dim + self.v_head_dim,
    )
    if self.q_lora_rank is None:
        query_states = self.q_proj(hidden_states)
    else:
        query_states = self.q_b_proj(
            self.q_a_layernorm(self.q_a_proj(hidden_states))
        )
    query_states = query_states.view(query_shape).transpose(1, 2)
    query_pass, query_rot = torch.split(
        query_states,
        [self.qk_nope_head_dim, self.qk_rope_head_dim],
        dim=-1,
    )

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    key_pass, key_rot = torch.split(
        compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
    )
    key_pass = self.kv_b_proj(self.kv_a_layernorm(key_pass))
    key_pass = key_pass.view(key_shape).transpose(1, 2)
    key_pass, value_states = torch.split(
        key_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
    )
    key_rot = key_rot.view(batch_size, 1, sequence_length, self.qk_rope_head_dim)

    cos, sin = position_embeddings
    if self.config.rope_interleave:
        query_rot, key_rot = apply_rotary_pos_emb_interleave(
            query_rot, key_rot, cos, sin
        )
    else:
        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)
    key_rot = key_rot.expand(*key_pass.shape[:-1], -1)
    query_states = torch.cat((query_pass, query_rot), dim=-1)
    key_states = torch.cat((key_pass, key_rot), dim=-1)

    if sequence_length > 1:
        _SANTAPP_PREFILL_QUERIES[self.layer_idx] = query_states.detach()
    if past_key_values is not None:
        cache_kwargs = {
            "cos": cos,
            "sin": sin,
            "cache_position": cache_position,
        }
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    recent_window = getattr(self.config, "santapp_recent_window", 64)
    use_sparse = (
        self.config.mode != "dense"
        and sequence_length == 1
        and key_states.shape[-2] > recent_window
    )
    if use_sparse:
        attention_output, attention_weights = santapp_attn(
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.scaling,
            mode=self.config.mode,
            sample_count=getattr(self.config, "santapp_samples_per_head", 128),
            recent_window=recent_window,
            group_size=getattr(self.config, "santapp_group_size", 16),
            cluster_feature=getattr(
                self.config, "santapp_cluster_feature", "fingerprint"
            ),
            probe_count=getattr(self.config, "santapp_probe_count", 64),
            probe_start_fraction=getattr(
                self.config, "santapp_probe_start_fraction", 0.75
            ),
            cluster_seed=getattr(self.config, "santapp_cluster_seed", 0),
            kmeans_iterations=getattr(
                self.config, "santapp_kmeans_iterations", 100
            ),
            summary_key=self.layer_idx,
            *args,
            **kwargs,
        )
    else:
        attention_output = sdpa(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attention_mask is None and sequence_length > 1,
            scale=self.scaling,
        )
        attention_weights = None

    attention_output = attention_output.transpose(1, 2)
    attention_output = attention_output.reshape(
        batch_size, sequence_length, -1
    ).contiguous()
    return self.o_proj(attention_output), attention_weights


def clear_attention_state() -> None:
    """Clear notebook attention summaries, boundaries and prefill queries."""
    _SANTAPP_SUMMARIES.clear()
    _SANTAPP_BOUNDARIES.clear()
    _SANTAPP_PREFILL_QUERIES.clear()


def patch_youtu_attention(model: YoutuPreTrainedModel) -> None:
    """Install notebook-compatible attention on every Youtu decoder layer."""
    clear_attention_state()
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(
            _youtu_attention_forward, layer.self_attn
        )


def unpatch_youtu_attention(model: YoutuPreTrainedModel) -> None:
    """Restore class-defined Youtu attention and clear runtime state."""
    for layer in model.model.layers:
        layer.self_attn.__dict__.pop("forward", None)
    clear_attention_state()
