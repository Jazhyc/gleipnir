# Completed full training-set diagnostic

`final-training-fit02` launched at **02:30:52 CEST on 2026-10-12** on existing
NC2 B200 `rzarg8j8x0so3i` and completed at **03:14:02 CEST**. Implementation:
`e2f494f`; ten focused tests and lint passed. No new capacity, training or
lifecycle operation occurred. Both fixed final checkpoints scored all 8,688
original training rows and **83,816,369 input tokens** without truncation.

Final pooled CE/KL are **0.180095/0.023103 for 4B**, versus
**0.179123/0.022131 for 9B**, with identical teacher entropy 0.156991554 nats.
The 9B pooled KL advantage is **4.2%**, smaller than the 13.5% difference in
online epoch summaries. Most of its advantage is on Bash Bench; Gloom is
slightly worse. This does not establish a hard capacity ceiling. Complete
source/label results, timing, numerical gates and interpretation are in the
[findings](../../docs/findings/monitor_training_loss_fit.md).

Both adapters passed master/merge, master/serving and merged/serving gates,
with actual BF16 backbone/native attention and FP32 two-row head dispatch.
The independent local completion audit verifies all **17,376 raw predictions**,
original teacher/label identities, reconstructed rendered prompts, paired token
counts, input/prediction hashes and full CE/entropy/KL/MAE recomputation.
Original master/export weights are preserved; only the final 9B merge remains
in the shared ephemeral slot. The owned scorers retired, with GPU utilization
0% and allocation 0 MiB at the subsequent live check. The Pod remains running.
Shutdown logged a 4B EngineDeadError and semaphore cleanup warnings after
complete scoring; no predictions failed or were excluded.

The first preparation caught a display-source versus dataset-key mismatch
before any model/GPU scoring. Its incomplete preparation and failure receipt
are preserved. `final-training-fit01` then stopped before merging/scoring
because historical completion receipts lack modern master/export SHA fields.
The previous disposable partial-AUC merge was retired only after verifying its
manifest and persistent master/export. Attempt 02 preserves both failures,
verifies original master hashes against the saved parity references, and writes
normalized receipts beside read-only links to original adapters. Complete 01
workloads were reused after input, target, teacher-prompt and identity checks.
No checkpoint, data, precision or objective selection changed.

Queue: `results/final-training-fit02/status.json` and `comparison.json`.
Each model's raw predictions and receipts:
`results/final-training-fit02-{4b,9b}/`; final metrics: `fit.json` there.
The reusable evaluator's storage key `evaluation/id.jsonl` contains
**training-seen examples**, not held-out ID evaluation. Logs:
`logs/runpod/final-training-fit02*/`. Frozen contracts, normalized receipts,
raw predictions, logs and failed-startup evidence were collected locally without
weights or full prompt copies. The detached queue completed without an in-chat
heartbeat scheduler.
