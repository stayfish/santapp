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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from .mla_benchmarking import DEFAULT_TASKS
    from .santapp import (
        patch_deepseek_v2_attention,
        patch_youtu_attention,
        unpatch_deepseek_v2_attention,
        unpatch_youtu_attention,
    )
except ImportError:
    from mla_benchmarking import DEFAULT_TASKS
    from santapp import (
        patch_deepseek_v2_attention,
        patch_youtu_attention,
        unpatch_deepseek_v2_attention,
        unpatch_youtu_attention,
    )


MODEL_ID = "tencent/Youtu-LLM-2B"


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
    parser.add_argument(
        "--architecture",
        choices=("youtu", "deepseek-v2"),
        default="youtu",
        help="Select the model-specific MLA projection and RoPE adapter.",
    )
    parser.add_argument(
        "--mode",
        choices=("dense", "topk", "guided", "santa", "uniform"),
        default="guided",
    )
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--cluster-size", type=int, default=16)
    parser.add_argument("--token-budget", type=int, default=128)
    parser.add_argument("--recent-window", type=int, default=64)
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load NF4 weights for 32GB GPUs such as V100.",
    )
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

    # Guided/SANTA sampling uses torch.multinomial; keep runs reproducible.
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    RULER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(
        RULER_DATA_DIR / "huggingface" / "datasets"
    )
    if "ruler_qa_hotpot" in args.tasks:
        install_hotpot_loader(args.hotpot_data)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=model_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=model_dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map={"": "cuda:0"} if args.load_in_4bit else None,
    )
    if not args.load_in_4bit:
        model = model.to("cuda")
    model = model.eval()
    model.config.mode = args.mode
    model.config.santapp_samples_per_head = args.token_budget
    model.config.santapp_recent_window = args.recent_window
    model.config.santapp_group_size = args.cluster_size
    model.config.santapp_cluster_feature = "key"
    model.config.santapp_probe_count = 64
    model.config.santapp_probe_start_fraction = 0.75
    model.config.santapp_cluster_seed = 0
    model.config.santapp_kmeans_iterations = args.kmeans_iterations
    if args.architecture == "deepseek-v2":
        patch_attention = patch_deepseek_v2_attention
        unpatch_attention = unpatch_deepseek_v2_attention
    else:
        patch_attention = patch_youtu_attention
        unpatch_attention = unpatch_youtu_attention
    patch_attention(model)

    lm = HFLM(
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
                "cluster_algorithm": "notebook_minibatch",
                "cluster_feature": model.config.santapp_cluster_feature,
                "attention_mode": args.mode,
                "torch_seed": 0,
                "architecture": args.architecture,
                "dtype": args.dtype,
                "load_in_4bit": args.load_in_4bit,
            },
        )
        if results is None:
            raise RuntimeError("lm-eval returned no results.")
        samples = results.get("samples", {})
        tracker.save_results_aggregated(results=results, samples=samples)
        for task_name, task_samples in samples.items():
            tracker.save_results_samples(task_name, task_samples)
    finally:
        unpatch_attention(model)

    print(f"Results written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
