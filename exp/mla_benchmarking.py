"""Run six RULER tasks at 8K context, using 20 prompts per task by default.

Install the benchmark dependency with ``pip install "lm-eval[datasets]"``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(os.environ.get("BENCHMARK_ROOT", Path(__file__).resolve().parents[1]))
RULER_DATA_DIR = REPO_ROOT / "data" / "ruler"
DEFAULT_TASKS = (
    "niah_single_2",      # single-needle retrieval
    "niah_multikey_2",    # multi-key retrieval
    "niah_multiquery",    # multi-query retrieval
    "ruler_vt",           # multi-hop variable tracking
    "ruler_cwe",          # common-word aggregation
    "ruler_qa_hotpot",    # question answering
)


def parse_args() -> argparse.Namespace:
    """Parse model choices while keeping the default evaluation reproducible."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V2-Lite",
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument("--backend", default="hf", help="lm-eval backend, e.g. hf or vllm.")
    parser.add_argument(
        "--model-args",
        default="trust_remote_code=True,dtype=bfloat16",
        help="Additional comma-separated lm-eval model arguments.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "ruler_8k")
    parser.add_argument("--apply-chat-template", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    """Build the lm-eval command for prompts at the requested context length."""
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive")

    model_args = f"pretrained={args.model},max_length={args.context_length}"
    if args.model_args:
        model_args += f",{args.model_args}"

    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        args.backend,
        "--model_args",
        model_args,
        "--tasks",
        ",".join(args.tasks),
        "--metadata",
        json.dumps({"max_seq_lengths": [args.context_length]}),
        "--limit",
        str(args.num_prompts),
        "--batch_size",
        args.batch_size,
        "--device",
        args.device,
        "--output_path",
        str(args.output),
        "--log_samples",
    ]
    if args.apply_chat_template:
        command.append("--apply_chat_template")
    return command


def main() -> None:
    """Run RULER and keep all downloaded/generated data inside this repository."""
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    RULER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = str(RULER_DATA_DIR / "huggingface")
    env["HF_DATASETS_CACHE"] = str(RULER_DATA_DIR / "huggingface" / "datasets")
    subprocess.run(build_command(args), check=True, env=env)


if __name__ == "__main__":
    main()
