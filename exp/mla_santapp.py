"""Local-GPU comparison of dense and clustered attention on Youtu-LLM-2B.

Requires ``torch`` and ``transformers>=5.1``. The experiment is intentionally
batch-size one: prefill stays dense, while incremental decoding can read only
the most relevant clusters plus a recent exact window.
"""

from __future__ import annotations

import argparse
import math
import types
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "tencent/Youtu-LLM-2B"


@dataclass
class ApproximateConfig:
    """Controls clustered attention during one-token decoding."""

    mode: str = "dense"
    cluster_size: int = 16
    token_budget: int = 128
    recent_window: int = 64
    kmeans_iterations: int = 20
    seed: int = 0


class YoutuApproximateAttention:
    """Runtime attention replacement specialized for batch-one Youtu MLA."""

    def __init__(self, model, config: ApproximateConfig):
        self.model = model
        self.config = config
        self.kv_cache: dict[int, list[torch.Tensor]] = {}
        self.summaries: dict[tuple[int, int], dict[str, object]] = {}
        self.cluster_end = 0

    @staticmethod
    def _interleaved_rope(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply Youtu interleaved RoPE to ``x`` shaped [B, H, T, D]."""
        cos = cos[..., : cos.shape[-1] // 2].unsqueeze(1)
        sin = sin[..., : sin.shape[-1] // 2].unsqueeze(1)
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        return torch.cat(
            (x_even * cos - x_odd * sin, x_odd * cos + x_even * sin),
            dim=-1,
        )

    @staticmethod
    @torch.inference_mode()
    def _kmeans(
        x: torch.Tensor, n_clusters: int, iterations: int, seed: int
    ) -> torch.Tensor:
        """Cluster input ``x`` with shape [tokens, features] on its CUDA device."""
        if x.ndim != 2:
            raise ValueError(f"Expected [tokens, features], got {tuple(x.shape)}")
        n_clusters = min(max(1, n_clusters), x.shape[0])
        generator = torch.Generator(device=x.device).manual_seed(seed)
        centers = x[
            torch.randperm(x.shape[0], generator=generator, device=x.device)[
                :n_clusters
            ]
        ].clone()
        labels = torch.full(
            (x.shape[0],), -1, dtype=torch.long, device=x.device
        )
        for _ in range(iterations):
            distances = torch.cdist(x, centers)
            new_labels = distances.argmin(dim=1)
            if torch.equal(new_labels, labels):
                break
            labels = new_labels
            counts = torch.bincount(labels, minlength=n_clusters)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, labels, x)
            active = counts > 0
            centers[active] = sums[active] / counts[active, None]
        return labels

    @torch.inference_mode()
    def build_summaries(self) -> None:
        """Cluster the old prompt keys after a dense prefill."""
        if not self.kv_cache:
            raise RuntimeError("Run a dense prefill before build_summaries().")
        prompt_length = next(iter(self.kv_cache.values()))[0].shape[1]
        self.cluster_end = max(0, prompt_length - self.config.recent_window)
        self.summaries.clear()
        if self.cluster_end == 0:
            return

        for layer, (keys, _) in self.kv_cache.items():
            for head in range(keys.shape[0]):
                old_keys = keys[head, : self.cluster_end].float()
                n_clusters = max(
                    1,
                    math.ceil(self.cluster_end / self.config.cluster_size),
                )
                labels = self._kmeans(
                    old_keys,
                    n_clusters,
                    self.config.kmeans_iterations,
                    self.config.seed + layer * keys.shape[0] + head,
                )
                groups = [
                    torch.where(labels == cluster)[0]
                    for cluster in range(n_clusters)
                ]
                groups = [group for group in groups if group.numel()]
                self.summaries[(layer, head)] = {
                    "indices": groups,
                    "sizes": torch.tensor(
                        [group.numel() for group in groups],
                        device=keys.device,
                        dtype=torch.float32,
                    ),
                    "centers": torch.stack(
                        [old_keys[group].mean(dim=0) for group in groups]
                    ),
                }

    def approximate(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer: int,
        head: int,
        scaling: float,
    ) -> torch.Tensor:
        r"""Estimate one attention head for query shape [D].

        Dense attention is
        \(y=\sum_i \exp(s_i)v_i / \sum_i\exp(s_i)\), where
        \(s_i=\mathrm{scaling}\,q^\top k_i\). ``topk`` ranks key clusters
        by \(\log |G|+\mathrm{scaling}\,q^\top\bar{k}_G\), reads complete
        clusters until the token budget is reached, and always reads the
        recent window exactly.
        """
        if self.config.mode == "dense" or self.cluster_end == 0:
            scores = keys.float() @ query.float() * scaling
            return torch.softmax(scores, dim=0) @ values.float()
        if self.config.mode != "topk":
            raise ValueError(f"Unknown approximate mode: {self.config.mode}")

        summary = self.summaries[(layer, head)]
        group_scores = (
            summary["centers"] @ query.float() * scaling
            + summary["sizes"].log()
        )
        selected: list[torch.Tensor] = []
        count = 0
        for group_id in group_scores.argsort(descending=True).tolist():
            group = summary["indices"][group_id]
            selected.append(group)
            count += group.numel()
            if count >= self.config.token_budget:
                break

        old_indices = torch.cat(selected)
        recent_indices = torch.arange(
            self.cluster_end, keys.shape[0], device=keys.device
        )
        indices = torch.cat((old_indices, recent_indices))
        scores = keys[indices].float() @ query.float() * scaling
        return torch.softmax(scores, dim=0) @ values[indices].float()

    def _forward(
        self,
        module,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        """Youtu attention forward for input shape [1, tokens, hidden_size]."""
        batch, tokens, _ = hidden_states.shape
        if batch != 1:
            raise NotImplementedError("Approximate attention supports batch size 1.")

        if module.q_lora_rank is None:
            query = module.q_proj(hidden_states)
        else:
            query = module.q_b_proj(
                module.q_a_layernorm(module.q_a_proj(hidden_states))
            )
        query = query.view(
            batch, tokens, module.num_heads, module.qk_head_dim
        ).transpose(1, 2)
        query_pass, query_rot = torch.split(
            query,
            [module.qk_nope_head_dim, module.qk_rope_head_dim],
            dim=-1,
        )

        compressed = module.kv_a_proj_with_mqa(hidden_states)
        key_pass, key_rot = torch.split(
            compressed,
            [module.kv_lora_rank, module.qk_rope_head_dim],
            dim=-1,
        )
        key_pass = module.kv_a_layernorm(key_pass)
        key_rot = key_rot.view(
            batch, 1, tokens, module.qk_rope_head_dim
        )
        cos, sin = position_embeddings
        query_rot = self._interleaved_rope(query_rot, cos, sin)
        key_rot = self._interleaved_rope(key_rot, cos, sin)
        query = torch.cat((query_pass, query_rot), dim=-1)
        if hasattr(module, "expand_kv"):
            keys, values = module.expand_kv(key_pass, key_rot)
        else:
            key_shape = (
                batch,
                tokens,
                module.num_heads,
                module.qk_nope_head_dim + module.v_head_dim,
            )
            key_pass = module.kv_b_proj(key_pass).view(key_shape)
            key_pass = key_pass.transpose(1, 2)
            key_pass, values = torch.split(
                key_pass,
                [module.qk_nope_head_dim, module.v_head_dim],
                dim=-1,
            )
            key_rot = key_rot.expand(*key_pass.shape[:-1], -1)
            keys = torch.cat((key_pass, key_rot), dim=-1)

        layer = module.layer_idx
        if layer in self.kv_cache:
            self.kv_cache[layer][0] = torch.cat(
                (self.kv_cache[layer][0], keys[0]), dim=1
            )
            self.kv_cache[layer][1] = torch.cat(
                (self.kv_cache[layer][1], values[0]), dim=1
            )
        else:
            self.kv_cache[layer] = [
                keys[0].contiguous(),
                values[0].contiguous(),
            ]
        full_keys, full_values = self.kv_cache[layer]

        if tokens > 1 or self.config.mode == "dense":
            output = F.scaled_dot_product_attention(
                query,
                full_keys.unsqueeze(0),
                full_values.unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=tokens > 1,
                scale=module.scaling,
            ).transpose(1, 2)
        else:
            output = torch.stack(
                [
                    self.approximate(
                        query[0, head, 0],
                        full_keys[head],
                        full_values[head],
                        layer,
                        head,
                        module.scaling,
                    )
                    for head in range(module.num_heads)
                ]
            )[None, None].to(hidden_states.dtype)

        output = output.reshape(batch, tokens, -1).contiguous()
        return module.o_proj(output), None

    def patch(self) -> None:
        """Install the replacement on all Youtu attention layers."""
        for layer in self.model.model.layers:
            layer.self_attn.forward = types.MethodType(
                lambda module, *args, _owner=self, **kwargs: _owner._forward(
                    module, *args, **kwargs
                ),
                layer.self_attn,
            )

    def unpatch(self) -> None:
        """Restore class-defined Hugging Face attention forwards."""
        for layer in self.model.model.layers:
            layer.self_attn.__dict__.pop("forward", None)
        self.kv_cache.clear()
        self.summaries.clear()

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, new_tokens: int) -> list[int]:
        """Greedily generate from input shape [1, prompt_tokens]."""
        self.kv_cache.clear()
        self.summaries.clear()
        original_mode = self.config.mode
        self.config.mode = "dense"
        output = self.model(input_ids, use_cache=False)
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = [int(next_token.item())]
        self.build_summaries()
        self.config.mode = original_mode

        position = input_ids.shape[1]
        for _ in range(new_tokens - 1):
            output = self.model(
                next_token,
                position_ids=torch.tensor(
                    [[position]], device=input_ids.device
                ),
                use_cache=False,
            )
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))
            position += 1
        return generated


def parse_args() -> argparse.Namespace:
    """Parse the small local comparison experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--prompt", default="Explain why the sky is blue.")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--cluster-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=128)
    parser.add_argument("--recent-window", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    """Load Youtu on one local GPU and compare dense with approximate output."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    repeated_prompt = (args.prompt.strip() + "\n") * max(
        1, args.prompt_tokens
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": repeated_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.prompt_tokens,
    ).input_ids.to("cuda")

    with torch.inference_mode():
        stock_output = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=args.new_tokens,
            do_sample=False,
        )
    stock_ids = stock_output[0, input_ids.shape[1] :].tolist()

    config = ApproximateConfig(
        cluster_size=args.cluster_size,
        token_budget=args.token_budget,
        recent_window=args.recent_window,
    )
    experiment = YoutuApproximateAttention(model, config)
    experiment.patch()
    try:
        config.mode = "dense"
        dense_ids = experiment.generate(input_ids, args.new_tokens)
        config.mode = "topk"
        approximate_ids = experiment.generate(input_ids, args.new_tokens)
    finally:
        experiment.unpatch()

    print("GPU:", torch.cuda.get_device_name(0))
    print("prompt tokens:", input_ids.shape[1])
    print("stock:", tokenizer.decode(stock_ids, skip_special_tokens=True))
    print("dense:", tokenizer.decode(dense_ids, skip_special_tokens=True))
    print(
        "approximate:",
        tokenizer.decode(approximate_ids, skip_special_tokens=True),
    )
    dense_matches = sum(a == b for a, b in zip(stock_ids, dense_ids))
    approximate_matches = sum(
        a == b for a, b in zip(dense_ids, approximate_ids)
    )
    print(f"stock/dense-patch agreement: {dense_matches}/{len(dense_ids)}")
    print(
        "dense/approximate agreement:",
        f"{approximate_matches}/{len(dense_ids)}",
    )


if __name__ == "__main__":
    main()
