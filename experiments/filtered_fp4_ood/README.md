# Strict-filtered FP4 MLP/FA4 OOD evaluation

The user requests OOD scoring of the original strict-filtered FP4 MLP/BF16
FlashQLA/FA4-trained adapter, selecting BF16 serving only. Optimized serving is
reserved for QAD-adapted candidates. This is a fixed-checkpoint transfer diagnostic,
not retraining, checkpoint selection, prompt tuning or model promotion.

Restore the original `filtered-fp4-fa4-baseline01` FP32 master (SHA256
`4eb97e5385f53b135c0b77cb92d0cbbfa78a661100f9ba8ac4205dbabb34f057`).
Reuse its original inference export and matched master/export parity; merge in
FP32 and export BF16 into the single
owned ephemeral slot, verifying the prior QAD64 merge before replacement. Require
exact original merged weight/tokenizer/config hashes, reuse its checksum-bound
master reference and perform three fresh BF16 serving canaries with nonzero
adapter effect and a passing BF16 dispatch audit before full scoring.

Score two fixed c128 passes over all 6,395 unchanged OOD examples. Reuse the
source and rendered workloads from `teacher-disagreement-strict-ood01`: exact
IDs, labels, source/rendered prompt hashes, token lengths and source populations.
Retain the frozen missing-prose/input contract for paper comparability; no prose
restoration or input conversion. Verify no exact trajectory overlap with the
original strict-filtered training rows. Higher-level lineage independence remains
unestablished, and this OOD suite has been evaluated repeatedly.

Primary comparison is the cached strict-filtered BF16/SDPA model; also report
cached released Gleipnir 4B and regular unfiltered scores. Rendering and labels
must match exactly; historical training/serving/runtime/host differences prevent
isolated causal claims. No new controls or teacher calls. Report equal-source raw
normalized pAUROC@20, pooled/per-source AUROC, calibration, fixed-threshold FPR/
recall, ties, repeated scores and input tokens/s. Both fixed passes are descriptive;
there is no selection/promotion rule based on OOD scores.

Stop on source/input/master/merge/GPU drift, failed parity or BF16 audit, missing/
nonfinite scores, truncation, backend failures or incomplete coverage. Preserve
failed receipts without automatic precision fallback. Run on the existing NC2
B200 with persistent caches; retire the scorer at completion and leave capacity
running. Entrypoint: `python -m experiments.filtered_fp4_ood.launch`. Artifacts:
`results/filtered-fp4-ood01/`; logs: `logs/runpod/filtered-fp4-ood01/`.
No in-chat scheduler is available; check startup through advancing scores during
the active turn without promising later agent wakeups.
