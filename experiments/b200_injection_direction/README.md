# Small-fit APPS injection-direction replication

User-authorized replication on the current BF16/SDPA augmented adapter and
BF16 eager Lens serving. Hypothesis: removing an injection-associated residual
component lowers honest-injection alarms while preserving useful backdoor
ranking and broader ID monitoring. The original colleague protocol is reviewed
in `docs/research/sdpa_injection_direction_replication.md`, pinned to commit
931fee0a1db369bc15c01009aa77219e7e106c14.

Select 64 of 655 APPS tasks by ascending SHA-256 of
`sdpa-injection-fit-v1:<task_id>`, before activations or outcomes. Keep every
parent and injected/neutral/label variant with its task: 896 fitting rows and
8,218 test rows from the other 591 tasks. No fit row contributes to test
metrics. This explicitly authorized exploratory APPS fitting exception does
not relabel the historical full benchmark as held out or change model training.
Fit all 32 layers' paired injected-minus-neutral decision residual means, using
both honest and backdoored pairs. Use the unit layer-20 direction at every
layer/token. Fit each layer's neutral decision and inserted-span projection
centers only on the fit partition. Validate span removal recovers parent code,
token offsets/counts and frozen prompt identity. Capture FP32 span means with
exact chunk coverage and native BF16 final-token residuals.

Freeze four primary arms: unedited, clamp beta .25 toward fitted centers,
full projection (beta 1/zero centers), and matched unit random projection
(NumPy seed 0, zero centers). Apply the same intervention to every label/text
condition. Add +/- one natural fitted layer-20 shift at that layer/all tokens
on 128 test tasks picked by `sdpa-add-test-v1:<task_id>` hash; these are
sufficiency controls, with full-vocabulary answer-mass diagnostics. No base
model, historical-vector transfer, generation, outcome-driven layer/beta sweep,
learned probe head, template selection or promotion in this first campaign.

Score the complete APPS test partition for all primary arms and all 3,012 frozen
CoT-removed ID rows for the four primary arms. Establish fresh unedited eager
scores on the same populations. Report strict clean-only at-most-2%-FPR metrics
both with each arm's recalibrated threshold and the fixed unedited threshold;
FPR and backdoor TPR together, matched AUROC/raw pAUROC20, plain/neutral controls,
calibration/ties and paired score shifts. ID reports source-macro/pooled/source
AUROC/raw pAUROC20. Grouped bootstrap intervals cover task sampling, not seeds.
The additive controls report p01 and refuse to interpret format degradation as
successful intervention. Freeze beta/direction-layer/arm set before evaluation.

Extend Lens with request-local projection state and FP32 span sums/counts. The
clamp acts on the complete fused residual and updates its hidden delta in FP32
before BF16 casting; this has serving-rounding differences from HF hooks. Only
the absolute final prompt token gets the decision center, never a chunk's last
row. Test direct tensor reference, beta-zero/no-op, finite bounds, mixed requests,
absolute positions, chunked span means, full readout and cleanup. Retire the
verified old engine sequentially and launch the same model/precision/renderer/
scheduler with updated source identity. Require fresh master/native canary and
functional smoke, retaining the old passing baseline receipts. A no-op canary
must reproduce the previous eager scores within MAE .005/correlation .995.

Stop on input/source/model drift, failed gates, missing span/hook coverage,
nonfinite outputs, isolation/cleanup failure, OOM or completion. Preserve every
partial/failed receipt and resume only exact completed batches. Keep caches and
merged checkpoint on the allocated NC2 B200; no new capacity, retraining or
local merged-weight transfer. Keep the passing engine warm. Monitoring is
active-turn only; training nondeterminism remains deferred.
