#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

.venv-exp/bin/python exp/mla_santapp_benchmark.py \
    --mode topk \
    --context-length 8192 \
    --num-prompts 20 \
    --cluster-size 16 \
    --token-budget 128 \
    --recent-window 64 \
    --kmeans-iterations 20 \
    --tasks \
        niah_single_2 \
        niah_multikey_2 \
        niah_multiquery \
        ruler_vt \
        ruler_cwe \
        ruler_qa_hotpot \
    --output results/ruler_8k_youtu_approximate
