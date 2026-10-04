# Monitor-directed injection augmentation: Qwen3.5-9B replication

Authorized 2026-10-04: replicate the completed 4B intervention on fresh
Qwen3.5-9B at 5e-5, then collect and verify all artifacts before shutting down
the existing Runpod B200. No additional capacity or publication is requested.

Training completed 2026-10-04: all 272 updates at 5e-5, with the expected fresh
initialization and matched metadata checks. The master, serving weights and final
checkpoint are collected and their completion-receipt checksums verify locally.
Bounded reference scoring finished; serving parity and full evaluation remain
pending. See the [finding](../../docs/findings/monitor_injection_augmentation_9b.md).

Hypothesis: the same label-independent monitor-directed distractors reduce
injection-only alarms at 9B while preserving clean behavior discrimination.
Reuse exactly the frozen 8,688 training rows from
`data/monitor_injection_augmentation/`: 3,475 replaced views, 5,213 clean,
the same eight training payload templates, role/position/demand assignments,
regular instruction and original Kimi soft BCE targets. Preserve all original
source bytes and provenance. No regeneration or preference task is introduced.
No held-out APPS benchmark inputs or attack payloads are added to training;
the original training population remains unchanged.

Train one epoch, 272 updates, seed 0, fresh rank-128/alpha-256 LoRA,
AdamW 5e-5, linear decay and 3% warmup. Reuse the completed 9B regular run's
BF16 FlashQLA/SDPA, 16,384-token best-fit packing, logical batch 32, physical
max-8 policy and FP32 masters. Its all-layer nonreentrant checkpointing and
checkpointed shell compilation accommodate the larger model. This is the
previously validated 9B adaptation of the 4B recipe, not a new systems sweep.
Reuse its strict packing and longest-input receipts with explicit skipped
markers; preserve failed broader strict parity results. Keep identity/input
checks and finite/missing-gradient checks. Reuse the shared persistent compiler
cache, allowing automatic compilation of new shapes.

Compare the fixed final checkpoint against cached original 9B regular results;
do not rescore the original adapter. Score the same 5,184 APPS benchmark,
3,930 honest controls, 3,012 canonical clean ID and 3,012 paired held-out injected
ID views with the regular prompt only. The synthetic template families and
source-lineage holdouts remain exactly as in the 4B experiment. A bounded
20-example base/master reference and master/serving parity gate precede full
scoring in one persistent Triton-GDN vLLM engine. No APPS-based selection.

Report source-macro project pAUROC@20, AUROC, calibration, ties, source/role/
family/demand-alignment effects, clean-calibrated injected-harmless FPR and
harmful TPR, and paired score shifts/flips. The historical synthetic `pauc20`
field is sklearn-standardized; separately report the project-normalized partial
area for matching clean/injected comparisons. There is no cached original
synthetic baseline. This remains one seed with a small off-policy template bank.

Stop on changed frozen inputs/code/targets, truncation/context overflow,
initialization or recipe drift, missing/nonfinite gradients, OOM, failed serving
parity, or incomplete score coverage. Preserve failure receipts and recover
implementation/infrastructure faults without changing the frozen intervention.
Do not shut down until master/serving weights, raw scores, manifests and logs
are collected and checksummed locally, results are recomputed/audited and
documented, and remote checks confirm no unfinished authorized workload.

Entrypoints: `prepare`, `train`, `launch_evaluate`; remote sequential launcher
`run.sh`. Artifacts: `data/monitor_injection_augmentation_9b/`,
`results/monitor_injection_augmentation_9b/` and
`logs/runpod/monitor_injection_augmentation_9b/`. Only the latter campaign may
write its artifacts; preserve completed 4B files and prior 9B baseline receipts.
