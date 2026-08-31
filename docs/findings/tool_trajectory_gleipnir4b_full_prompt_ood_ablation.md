# Gleipnir 4B full-prompt OOD ablation

Status: completed and audited 2026-08-31.

## Question and frozen design

The mixed-data Gleipnir 4B adapter was trained on the compact student prompt
against Kimi K3 distributions produced with the full teacher monitoring
specification. This ablation asks whether the compact deployment instruction
under-elicits knowledge that the adapter already has.

The intervention replaces only the compact prompt with the frozen full teacher
prompt. It keeps the same completed rank-128, one-epoch, soft-only adapter; all
6,395 strict-OOD rows; Qwen3.5-4B model revision; non-thinking assistant
boundary; and normalized next-token logits for literal `0` and `1`. The full
teacher prompt contains the richer task specification but no row label or
other example-specific privileged information. No checkpoint, prompt, subset,
threshold, or stopping decision was selected on OOD labels.

The executable frozen config is
[`../../experiments/tool_trajectory_monitoring/distillation_mixed_qwen4b_teacher_prompt_ood_benchmark.json`](../../experiments/tool_trajectory_monitoring/distillation_mixed_qwen4b_teacher_prompt_ood_benchmark.json).
Its SHA-256 is
`35c99ae53e2577f2459b726c6e9ddd6d081946b9e20f0412a28660a8da94f323`.

## Result

| Prompt | Mean-OOD pAUROC@20 | Mean-OOD AUROC | Pooled pAUROC@20 | Pooled AUROC | Macro Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact student | 0.782350 | 0.908584 | 0.701531 | 0.913513 | 0.132055 |
| Full teacher | 0.782758 | 0.907718 | 0.713957 | 0.914140 | 0.130278 |
| Full minus compact | +0.000409 | -0.000865 | +0.012426 | +0.000628 | -0.001777 |

The primary source-balanced metric and regular AUROC are effectively unchanged.
The full prompt modestly improves pooled pAUROC and calibration, but pooled
metrics weight sources by row count and include cross-source ranking pairs.
They do not override the frozen source-balanced primary metric.

| OOD source | pAUROC@20 change | AUROC change |
| --- | ---: | ---: |
| AgentDojo | -0.019742 | -0.002663 |
| CUA-SHADE | -0.002531 | +0.003996 |
| IaC | +0.005335 | +0.000362 |
| MLE-Sabotage | +0.000000 | +0.000000 |
| Rogue Eval | +0.011958 | +0.003020 |
| SHADE-Arena | +0.007432 | -0.009907 |

The prompt does affect individual examples: full- and compact-prompt scores
have Pearson correlation `0.984396`, mean absolute difference `0.033225`, and
maximum absolute difference `0.648224`; 217 examples cross the 0.5 threshold.
Those changes offset across sources rather than producing an aggregate gain.

## Cost and audit

The full prompt uses 60,539,619 input tokens (mean 9,466.711; maximum 30,382),
versus 55,346,879 for the compact prompt. This is a 5,192,740-token or 9.38%
increase. At the existing `$0.03/M` input and `$0.15/M` output proxy, cost rises
from `$0.2598` to `$0.2842` per 1,000 examples.

The full run completed on the `gleipnir-control` NVIDIA H100 80 GB in 1,227.8
seconds. All 6,395 unique IDs match the compact run, every score is finite, the
maximum prompt remains below the 32,768-token engine limit, and no OOM or engine
error occurred. The existing master-checkpoint/serving parity gate passed before
launch, and a balanced 12-row canary showed a nonzero adapter effect. The result
and prediction SHA-256 values are respectively
`01b6b50c49664ca7086c5caca1c3ce4162e2aeacb2cc92d87158b119aebe6209` and
`dba806f3b9d840fffbab97016ef9cf32a9aecd692ac621aa622bac5bf5c59b81`.

## Interpretation

The evidence does not support the compact student prompt as a major aggregate
constraint on the mixed Gleipnir 4B adapter. The compact interface appears to
contain enough information to elicit essentially the same source-balanced OOD
ranking, while being cheaper.

This ablation does not estimate privileged-distillation noise directly. It
holds the trained adapter fixed, so it cannot separate noisy Kimi targets from
student capacity, optimization, or training-data limitations. A causal test of
privileged-context noise would require matched students trained from teacher
distributions produced under compact versus full prompts, with the student
evaluation prompt held fixed.
