"""Evaluate clustered Youtu attention on six 8K RULER tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RULER_DATA_DIR = REPO_ROOT / "data" / "ruler"
DEFAULT_HOTPOT_DATA = (
    RULER_DATA_DIR
    / "official"
    / "scripts"
    / "data"
    / "synthetic"
    / "json"
    / "hotpotqa.json"
)
HOTPOT_FALLBACK_URL = (
    "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/"
    "hotpot_dev_distractor_v1.json?download=true"
)
HOTPOT_OBSOLETE_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/"
    "hotpot_dev_distractor_v1.json"
)
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(RULER_DATA_DIR / "huggingface" / "datasets"),
)

import requests
import torch
from lm_eval import evaluator
from lm_eval.loggers import EvaluationTracker
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .mla_benchmarking import DEFAULT_TASKS
    from .mla_santapp import (
        MODEL_ID,
        ApproximateConfig,
        YoutuApproximateAttention,
    )
except ImportError:
    from mla_benchmarking import DEFAULT_TASKS
    from mla_santapp import (
        MODEL_ID,
        ApproximateConfig,
        YoutuApproximateAttention,
    )

class ApproximateYoutuHFLM(HFLM):
    """Use the batch-one custom generator for lm-eval generation requests."""

    def __init__(self, experiment: YoutuApproximateAttention, **kwargs):
        self.experiment = experiment
        super().__init__(**kwargs)

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: list[str],
        **generation_kwargs,
    ) -> torch.Tensor:
        """Generate from input ``context`` shaped [1, prompt_tokens]."""
        if context.shape[0] != 1:
            raise ValueError("Approximate Youtu benchmark requires batch size 1.")
        new_tokens = max_length - context.shape[1]
        if new_tokens <= 0:
            return context
        generated = self.experiment.generate(context, new_tokens)
        generated_tensor = torch.tensor(
            generated,
            dtype=context.dtype,
            device=context.device,
        )[None]
        return torch.cat((context, generated_tensor), dim=1)


def load_hotpotqa(path: Path) -> tuple[list[dict], list[str]]:
    """Load HotpotQA records from local JSON and build RULER document indices."""
    if path.exists():
        with path.open(encoding="utf-8") as hotpot_file:
            data = json.load(hotpot_file)
    else:
        from lm_eval.tasks.ruler.qa_utils import download_json

        print(
            f"HotpotQA file not found at {path}; using HTTPS fallback.",
            flush=True,
        )
        data = download_json(HOTPOT_FALLBACK_URL)

    documents = [
        f"{title}\n{''.join(paragraphs)}"
        for record in data
        for title, paragraphs in record["context"]
    ]
    documents = sorted(set(documents))
    document_ids = {
        document: index for index, document in enumerate(documents)
    }
    questions = [
        {
            "query": record["question"],
            "outputs": [record["answer"]],
            "context": [
                document_ids[f"{title}\n{''.join(paragraphs)}"]
                for title, paragraphs in record["context"]
            ],
        }
        for record in data
    ]
    return questions, documents


def install_hotpot_loader(path: Path) -> None:
    """Route lm-eval's HotpotQA task away from its obsolete HTTP URL."""
    original_get = requests.get

    if path.exists():
        with path.open(encoding="utf-8") as hotpot_file:
            local_data = json.load(hotpot_file)

        class LocalHotpotResponse:
            """Minimal requests response backed by the local official JSON."""

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> list[dict]:
                return local_data

        def redirected_get(url, *args, **kwargs):
            if url == HOTPOT_OBSOLETE_URL:
                return LocalHotpotResponse()
            return original_get(url, *args, **kwargs)

        source = path
    else:

        def redirected_get(url, *args, **kwargs):
            if url == HOTPOT_OBSOLETE_URL:
                url = HOTPOT_FALLBACK_URL
            return original_get(url, *args, **kwargs)

        source = HOTPOT_FALLBACK_URL

    requests.get = redirected_get
    print(f"HotpotQA source: {source}")


def parse_args() -> argparse.Namespace:
    """Parse reproducible six-task benchmark settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--mode", choices=("dense", "topk"), default="topk")
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--cluster-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=128)
    parser.add_argument("--recent-window", type=int, default=64)
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument(
        "--hotpot-data",
        type=Path,
        default=DEFAULT_HOTPOT_DATA,
        help="Local HotpotQA distractor JSON; HTTPS is used if it is absent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "ruler_8k_youtu_approximate",
    )
    parser.add_argument("--apply-chat-template", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the six RULER tasks with patched Youtu attention."""
    args = parse_args()
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    RULER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(
        RULER_DATA_DIR / "huggingface" / "datasets"
    )
    if "ruler_qa_hotpot" in args.tasks:
        install_hotpot_loader(args.hotpot_data)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    approximate_config = ApproximateConfig(
        mode=args.mode,
        cluster_size=args.cluster_size,
        token_budget=args.token_budget,
        recent_window=args.recent_window,
        kmeans_iterations=args.kmeans_iterations,
    )
    experiment = YoutuApproximateAttention(model, approximate_config)
    experiment.patch()

    lm = ApproximateYoutuHFLM(
        experiment=experiment,
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=1,
        max_length=args.context_length,
        device="cuda",
    )
    tracker = EvaluationTracker(output_path=str(args.output))
    try:
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=list(args.tasks),
            limit=args.num_prompts,
            batch_size=1,
            device="cuda",
            log_samples=True,
            evaluation_tracker=tracker,
            apply_chat_template=args.apply_chat_template,
            metadata={
                "max_seq_lengths": [args.context_length],
                "pretrained": args.model,
            },
        )
        if results is None:
            raise RuntimeError("lm-eval returned no results.")
        samples = results.get("samples", {})
        tracker.save_results_aggregated(results=results, samples=samples)
        for task_name, task_samples in samples.items():
            tracker.save_results_samples(task_name, task_samples)
    finally:
        experiment.unpatch()

    print(f"Results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
