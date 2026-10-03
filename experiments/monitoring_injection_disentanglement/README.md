# Injection presence and behavior disentanglement

Approved by the user on 2026-10-03: train one fresh Qwen3.5-4B adapter with the
reviewed behavior-grounded instruction, then evaluate on the prompt-injection
benchmark with neutral, aggressive and conservative instructions, plus the
frozen CoT-removed ID set as a sanity check. The mapping was explicitly
confirmed: neutral is the original regular instruction, aggressive is the
previous injection-aware instruction, and conservative is the approved draft.

Hypothesis: explicitly requiring task-behavior evidence reduces alarms driven
solely by injection text while preserving detection of underlying misconduct.
Intervention: change only the student instruction relative to the completed
one-epoch awareness campaign. Keep all 8,688 trajectories, source labels and
cached Kimi K3 soft targets, including the four assigned STRIDE objectives.
The fixed targets remain supervision from the earlier teacher rubric; this
tests a student-instruction intervention, not newly annotated labels.

Use pinned Qwen3.5-4B, the same zero-initialized rank-128/alpha-256 FP32 adapter,
seed 0, AdamW 5e-5, one epoch (272 logical updates), logical batch 32 and soft
BCE only. Use the selected packed BF16 B200 profile with 16,384-token physical
rows, intact long singletons, SDPA, pinned FlashQLA and no checkpointing.
Reuse the recorded unchanged-recipe validation with its checksum, explicitly
marking diagnostics as not performed; retain input/provenance audits and every
update's missing/nonfinite-gradient guards. A changed kernel/compiler, context
envelope or recipe requires new diagnostics rather than silent reuse.

Evaluate the final checkpoint; never select or tune prompts/thresholds against
these results. No source removal, teacher calls, attack generation or public
model upload is part of this campaign. The approved instruction is frozen in
`student_prompt.txt` and checked against the reviewed original during preparation.

Baseline weights are unadapted base, the previous regular-trained 4B and the
previous awareness-trained 4B. Extend their frozen two-prompt benchmark grid
with the conservative prompt and add the new adapter under all three prompts.
Reuse the six historical neutral/aggressive benchmark cells only after exact
prompt/membership/model checks, recording their existing contracts and hashes;
do not claim they were rescored. Score the six new benchmark cells and retain
all twelve cells without promoting a winner. The user-requested ID sanity check
scores only the new adapter with its conservative training instruction, once
over all 3,012 examples. The three-prompt sweep applies to the injection dataset.

The benchmark is the pinned APPS snapshot at HF revision
`58e722bf9912903d56a334f873e069aa17ccb7ad`: 5,184 original rows and 3,930
honest-code controls per instruction. Preserve original tasks, code, payloads,
labels and metadata when changing the monitor envelope. Honest controls are
essential: injected backdoors alone cannot distinguish code detection from
injection-text suspicion. ID has all 3,012 frozen CoT-removed rows, with
source/lineage holdouts checked against training. No final-test promotion.

Use original-FLA causal-master reference scores on 20 deterministic training
examples per prompt, then one persistent BF16 vLLM engine with continuous
batching, LoRA switching, the native non-thinking `Prediction:` boundary,
one constrained decision token and both requested raw decision logprobs.
Every new adapter/prompt combination must meet correlation >=0.99 and mean
absolute difference <=0.02, with a nonzero adapter effect, before full scoring.
Stop on identity drift, failed parity, missing/nonfinite scores or gradients,
truncation, OOM or incomplete coverage. Preserve failed receipts.

Report original per-template AUROC/TPR@2%FPR and injection-minus-neutral changes;
honest-injection alarms at each cell's original clean-honest threshold; and
matched AUROC when both classes carry the same injection. Preserve score shifts,
ties and ROC data. ID reports source-macro pAUROC@20/AUROC, calibration and fixed
threshold diagnostics. Labels measure code backdoors for the APPS benchmark,
and broad agent behavior for ID. One seed and fixed transfer templates do not
establish adaptive robustness or a causal role for the four STRIDE examples.

Artifacts: `data/monitoring_injection_disentanglement/`,
`results/monitoring_injection_disentanglement/` and
`logs/runpod/monitoring_injection_disentanglement/`. Use the existing authorized
B200 Pod `3422tdq7tfog52` at $6.79/hour and its preserved network volume.
Reuse the populated persistent compiler cache through the shared runtime helper.
The initial cold-cache attempt and its intentional warm-cache restart are
documented in [startup findings](../../docs/findings/monitoring_injection_disentanglement.md).
The current session has no in-chat scheduling tool: startup and campaign
monitoring happen in the active turn; no post-turn heartbeat is promised.

```bash
python -m experiments.monitoring_injection_disentanglement.prepare
python -m experiments.monitoring_injection_disentanglement.train
python -m experiments.monitoring_injection_disentanglement.evaluate --stage reference
python -m experiments.monitoring_injection_disentanglement.evaluate --stage vllm
```
