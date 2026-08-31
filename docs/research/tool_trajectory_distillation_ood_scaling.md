# Soft-distillation OOD data scaling

Status: frozen evaluation queued 2026-08-31; results pending.

This record isolates how strict-OOD monitoring changes as the amount of matched
tool-trajectory data increases under the inherited deception recipe. The
intervention is training-set size: `204`, `504`, `996`, `2,004`, `4,008`, and
`8,688` nested, source-label-balanced rows. Every point is one seed of the same
Qwen3.5-9B rank-128 QLoRA trained for exactly one epoch at learning rate `5e-5`,
effective batch size 32, and maximum sequence length 29,696. Supervision is only
Kimi K3's soft literal-token distribution; no arm uses hard labels or completion
loss.

The separate 21,837-row mixed condition adds 13,149 earlier deception rows to
all 8,688 matched tool-trajectory rows. It tests cross-domain transfer or
interference and is not part of the matched-data scaling fit. The same mixed
condition on Qwen3.5-4B is a backbone-size comparison under assumed transferred
hyperparameters.

## Frozen evaluation contract

All adapters are scored once on every row in the paper's six-source, 6,395-row
strict OOD suite. Evaluation uses the compact student prompt, native Qwen
thinking disabled, and normalized next-token logits for literal `0` versus `1`
after `Prediction:`. The primary metric is the unweighted mean across sources of
the paper's raw pAUROC through FPR 0.20, divided by 0.20. AUROC, pooled metrics,
calibration, fixed-threshold diagnostics, ties, per-source metrics, token counts,
runtime, and checksums are retained as secondary outputs.

The OOD labels cannot select prompts, subsets, checkpoints, thresholds, or stop
points. Every adapter is the final one-epoch checkpoint. Backend parity is gated
separately for the 9B and 4B layouts before full evaluation. Stop on parity
failure, adapter checksum drift, prompt truncation, missing decision-token
logprobs, non-finite scores, incomplete coverage, OOM, or another backend error.

The frozen executable contract is
[`../../experiments/tool_trajectory_monitoring/distillation_ood_benchmark.json`](../../experiments/tool_trajectory_monitoring/distillation_ood_benchmark.json).
Results and interpretation will be added here only after all eight adapter
evaluations complete and pass the artifact audit. The two full-mixed points will
also be added to
[`tool_trajectory_ood_frontier.md`](tool_trajectory_ood_frontier.md); the six
matched-size points remain here so the general frontier is not overloaded with
one experiment's internal scaling curve.
