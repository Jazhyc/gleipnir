# Projection after activation-based training removal

User-requested follow-up: apply the existing full projection to the completed
activation-ranked 20% removal model and measure APPS. Hypothesis: its increased
injection alarms still depend on the previously fitted residual direction.
This tests transfer of a fixed intervention across trained checkpoints.

Use the adapter-specific merged BF16 checkpoint and preserved FP32 master of
`activation-filter-ranked20-sdpa01`. Reuse its passing merge receipt, then run
fresh master/native/Lens/operator gates in eager BF16 vLLM 0.31/Lens 1.3.0.
This is the same mechanistic research exception to optimized serving as the
original direction campaign. Retire the verified resident scorer sequentially;
preserve all merges, weights, caches and receipts. No capacity lifecycle.

Freeze only two arms: fresh unedited eager scoring and full projection along
the original unit layer-20 injection direction, applied at all 32 layers/all
tokens with beta 1 and zero centers. Reuse the checksum-bound direction and
serialized operator exactly; no fitting, layer/beta selection, clamp, new random
control, retraining, ID evaluation or promotion in this follow-up.

Reuse all 8,218 rows from the original 591-task APPS test partition, excluding
all 64 direction-fitting tasks and their variants. Neither training removal nor
this evaluation makes that benchmark an untouched confirmation set. Score each
arm in the same warm eager engine with concurrency 64 and batches of 128. Reuse
the original fixed 128-task answer-mass diagnostic subset. Report clean-only
at-most-2%-FPR metrics using both each arm's threshold and the fixed unedited
threshold: injected-honest FPR, injected-backdoor recall, matched AUROC/raw
pAUROC20, vanilla/neutral controls, calibration, ties and score shifts. Compare
fresh unedited scores with the cached compiled-BF16 candidate as a serving-mode
diagnostic. Freeze arms/checkpoint/axis before scoring; single-seed transfer only.

Stop on identity/source/input drift, failed master/native/operator gates,
nonfinite scores, wrong membership, missing coverage, isolation/cleanup failure
or OOM. Preserve partial/failed receipts; resume only exact completed batches.
Keep a passing Lens engine warm, with weights remote. Monitoring is active-turn
only; no scheduling tool is available. `run.py` explicitly reuses the existing
SDPA Lens startup, injection operator smoke and resumable scorer; `startup.json`
binds the new adapter and `config.json` binds workloads/vector/baseline.
