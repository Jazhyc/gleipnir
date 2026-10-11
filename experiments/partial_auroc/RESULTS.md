# Completed run record

`strict-pauc02` launched at **00:14:41 CEST on 2026-10-12** on existing Runpod
B200 `rzarg8j8x0so3i` in NC2 and completed at **01:51:07 CEST**. Implementation:
`d6b1637`; 39 focused tests and lint passed. No new capacity or termination
occurred. Driver/scorer processes retired; the Pod remains running.

The final source-macro normalized pAUROC@20 is **0.872081 ID** and **0.795452
OOD**, versus **0.896106 / 0.787535** for the matched strict soft-BCE checkpoint
with FP32 head computation. OOD gains 0.7918 percentage points, while ID loses
2.4025 points. No default promotion occurred. The complete source/calibration/
threshold results and qualifications are in the
[findings](../../docs/findings/partial_auroc.md).

All 245 updates and all 7,836 training rows completed. The fresh gradient replay
canary passed with maximum parameter error `5.551115123125783e-17`; every native
score/replay margin difference was zero and all logged gradient norms were
finite. Both forward passes cover 75,851,675 tokens each, with 1,032.90 s spent
on the additional score passes. Master/merge/serving gates and the actual
BF16-backbone/FP32-head dispatch passed. All 9,407 matched predictions and
receipts were collected and independently audited locally; weights remain on
the persistent volume. Historical PIDs are not a live capacity inventory.

The earlier `strict-pauc01` attempt failed at offline tokenizer lookup because
the custom launcher omitted persistent HF cache paths. No model loaded and no
optimizer update occurred. Its launch/failure receipts and logs remain intact.
The corrected attempt changed cache/PATH setup and artifact name only; objective,
data, initialization, optimizer and selection settings were unchanged.

Receipts: `results/strict-pauc02/runner.json`, `partial_auc_training.json`,
`comparison.json` and `completion_audit.json`; logs:
`logs/runpod/strict-pauc02/`. The shared ephemeral merge slot contains only the
candidate; original strict master/export and the archived prior merge manifest
are preserved. The detached queue completed without an agent scheduler or
recurring agent follow-up.
