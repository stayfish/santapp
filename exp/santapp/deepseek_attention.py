"""Approximate attention adapter for the DeepSeek-V2-Lite MLA structure."""

from __future__ import annotations

import types
from typing import Any

import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa
from transformers.cache_utils import Cache
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2PreTrainedModel,
    apply_rotary_emb,
)

from .attention import (
    _SANTAPP_BOUNDARIES,
    _SANTAPP_PREFILL_QUERIES,
    _SANTAPP_SUMMARIES,
    clear_attention_state,
    santapp_attn,
)


def _deepseek_v2_attention_forward(
    self: DeepseekV2Attention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    position_embeddings: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""Run dense prefill or approximate decode for DeepSeek-V2 MLA.

    ``hidden_states`` enters with shape ``[B,T,hidden_size]``. DeepSeek first
    constructs per-head queries and expanded keys/values as

    .. math::
       q=[q^{C};q^{R}],\quad k=[k^{C};k^{R}],\quad
       q,k\in\mathbb{R}^{B\times H\times T\times D_q}.

    Prefill and dense mode use PyTorch SDPA. A one-token decode in any sparse
    mode delegates to :func:`santapp_attn`; the clustering implementation is
    therefore kept outside this attention adapter.
    """
    if hidden_states.ndim != 3:
        raise ValueError("Expected hidden_states=[B,T,hidden_size]")
    if position_embeddings is None:
        raise ValueError("DeepSeek-V2 attention requires rotary embeddings")

    batch_size, sequence_length = hidden_states.shape[:-1]
    if sequence_length > 1:
        _SANTAPP_BOUNDARIES.pop(self.layer_idx, None)
        _SANTAPP_PREFILL_QUERIES.pop(self.layer_idx, None)
        for key in [key for key in _SANTAPP_SUMMARIES if key[0] == self.layer_idx]:
            _SANTAPP_SUMMARIES.pop(key)

    query_shape = (batch_size, sequence_length, -1, self.qk_head_dim)
    key_value_shape = (
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
    query_nope, query_rope = torch.split(
        query_states,
        [self.qk_nope_head_dim, self.qk_rope_head_dim],
        dim=-1,
    )

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    compressed_keys, key_rope = torch.split(
        compressed_kv,
        [self.kv_lora_rank, self.qk_rope_head_dim],
        dim=-1,
    )
    expanded_kv = self.kv_b_proj(self.kv_a_layernorm(compressed_keys))
    expanded_kv = expanded_kv.view(key_value_shape).transpose(1, 2)
    key_nope, value_states = torch.split(
        expanded_kv,
        [self.qk_nope_head_dim, self.v_head_dim],
        dim=-1,
    )
    key_rope = key_rope.view(
        batch_size, 1, sequence_length, self.qk_rope_head_dim
    )
    query_rope, key_rope = apply_rotary_emb(
        query_rope, key_rope, position_embeddings.to(query_rope.device)
    )
    key_rope = key_rope.expand(*key_nope.shape[:-1], -1)

    query_states = torch.cat((query_nope, query_rope), dim=-1)
    key_states = torch.cat((key_nope, key_rope), dim=-1)
    if sequence_length > 1:
        _SANTAPP_PREFILL_QUERIES[self.layer_idx] = query_states.detach()

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx
        )

    mode = getattr(self.config, "mode", "dense")
    recent_window = getattr(self.config, "santapp_recent_window", 64)
    use_sparse = (
        mode != "dense"
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
            mode=mode,
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
            **kwargs,
        )
    else:
        attention_output = sdpa(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=attention_mask is None and sequence_length > 1,
            scale=self.scaling,
        )
        attention_weights = None

    attention_output = attention_output.transpose(1, 2)
    attention_output = attention_output.reshape(
        batch_size, sequence_length, self.num_heads * self.v_head_dim
    ).contiguous()
    return self.o_proj(attention_output), attention_weights


def patch_deepseek_v2_attention(model: DeepseekV2PreTrainedModel) -> None:
    """Install approximate attention on all DeepSeek-V2 decoder layers."""
    clear_attention_state()
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(
            _deepseek_v2_attention_forward, layer.self_attn
        )


def unpatch_deepseek_v2_attention(model: DeepseekV2PreTrainedModel) -> None:
    """Restore class-defined DeepSeek-V2 attention and clear runtime state."""
    for layer in model.model.layers:
        layer.self_attn.__dict__.pop("forward", None)
    clear_attention_state()
