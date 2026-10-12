# Initial verified launch

The fixed regular 4B/9B training-fit comparison is running as
`final-training-fit02` on existing NC2 B200 `rzarg8j8x0so3i`. The detached driver
launched at **02:30:52 CEST on 2026-10-12**, historical PID 8290. No new capacity,
training or lifecycle operation occurred. Both frozen model workloads contain
8,688 rows and **83,816,369 tokens**, matching the original training token audit.
The queue merges, checks and scores 4B, then 9B, using one disposable slot.

At the initial successful scoring check, 4B had passed master/merge parity
(MAE 0.00274321, correlation 0.99986627), master/serving parity
(MAE 0.00389763, correlation 0.99989203), and actual BF16-backbone/FP32-head
dispatch checks. The first 128 training examples completed successfully. Ten
focused tests and lint passed. This is startup evidence, not final training fit
or a completed 9B validation. Read live receipts for subsequent status.

The first preparation caught a display-source versus dataset-key mismatch
before any model/GPU scoring. Its incomplete preparation and failure receipt
are preserved. `final-training-fit01` then stopped before merging/scoring
because historical completion receipts lack modern master/export SHA fields.
The previous disposable partial-AUC merge was retired only after verifying its
manifest and persistent master/export. Attempt 02 preserves both failures,
verifies original master hashes against the saved parity references, and writes
normalized receipts beside read-only links to original adapters. Complete 01
workloads are reused after input, target, teacher-prompt and identity checks.
No checkpoint, data, precision or objective selection changed.

Queue: `results/final-training-fit02/status.json`. Each model's receipts and
predictions: `results/final-training-fit02-{4b,9b}/`; final metrics: `fit.json`
there and queue-level `comparison.json`. The storage key `evaluation/id.jsonl`
is used by the reusable evaluator; these are **training-seen examples**, not
held-out ID evaluation. Logs: `logs/runpod/final-training-fit02*/`.
Frozen manifests, normalized receipts, first parity/score receipts and failed
startup evidence were collected locally without weights or full prompt copies.
The process queue is unattended; no in-chat heartbeat scheduler is available.
The owned scorer retires at completion/failure, while the Pod remains running.
