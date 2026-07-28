# MLA Baseline Results

This document records the current `mla_benchmarking` baseline on the RULER
benchmark. RULER contains 13 subtasks in total. The experiments below evaluate
a six-subtask subset, so the results should not be reported as a complete
13-task RULER score.

Each subtask is scored from `0.0` to `1.0`, where `1.0` is the maximum score.

## Evaluated Tasks

| Task | RULER category | Description |
|---|---|---|
| `niah_single_2` | Retrieval | Single-needle retrieval from an essay haystack using a word key and numeric value; similar to vanilla NIAH. |
| `niah_multikey_2` | Retrieval | Multi-key retrieval; similar to key-value retrieval. |
| `niah_multiquery` | Retrieval | Retrieval of multiple requested needles or key-value pairs. |
| `ruler_vt` | Multi-hop tracing | Variable tracking across relationships in the context. |
| `ruler_cwe` | Aggregation | Common word extraction. |
| `ruler_qa_hotpot` | Question answering | Context-based question answering using HotpotQA. |

## Results

| Model | Experiment configuration | `niah_single_2` | `niah_multikey_2` | `niah_multiquery` | `ruler_cwe` | `ruler_vt` | `ruler_qa_hotpot` |
|---|---|---:|---:|---:|---:|---:|---:|
| Youtu-LLM-2B | 20 prompts/task, 8192 context, FP16 | 1.00 | 0.85 | 0.95 | 0.245 | 0.07 | 0.10 |
| Youtu-LLM-2B smoke | 2 prompts/task, 8192 context, FP16 | 1.00 | 1.00 | 1.00 | 0.05 | 0.10 | 0.00 |
| DeepSeek-V2-Lite smoke | 2 prompts/task, 4096 context, NF4 4-bit, one 32 GB V100 | 1.00 | 1.00 | 0.50 | 0.80 | 0.70 | 0.00 |

The Youtu-LLM-2B 20-prompt experiment is the current comparable baseline. It
evaluated 120 samples in total: 6 tasks with 20 prompts per task. The two smoke
tests contain only two prompts per task and are intended primarily to validate
the execution path. Their scores have high variance and should not be compared
directly with the 20-prompt baseline.

The DeepSeek smoke test also differs in context length and quantization, so it
is not a controlled model comparison against the Youtu baseline.

## Unevaluated RULER Tasks

The following seven official RULER subtasks have not yet been included:

| Task | Description |
|---|---|
| `niah_single_1` | Single-needle passkey retrieval from a repeated-text haystack. |
| `niah_single_3` | Single-needle retrieval with a UUID value. |
| `niah_multikey_1` | Multi-key line-retrieval variant. |
| `niah_multikey_3` | Additional multi-key retrieval variant. |
| `niah_multivalue` | Retrieval of multiple values associated with a key. |
| `ruler_fwe` | Frequent word extraction. |
| `ruler_qa_squad` | Context-based question answering using SQuAD v2. |

## Complete RULER Task Set

The complete 13-subtask RULER suite is:

```text
niah_single_1
niah_single_2
niah_single_3
niah_multikey_1
niah_multikey_2
niah_multikey_3
niah_multiquery
niah_multivalue
ruler_vt
ruler_cwe
ruler_fwe
ruler_qa_hotpot
ruler_qa_squad
```

Task definitions follow the
[lm-evaluation-harness RULER implementation](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/ruler)
and the [official NVIDIA RULER repository](https://github.com/NVIDIA/RULER).
