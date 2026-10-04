# Replicate monitor-directed injection augmentation on 9B

Date: 2026-10-04. User authorized the Qwen3.5-9B replication at 5e-5,
all required transfers to the existing Runpod B200, and shutdown after training,
evaluation and verified collection. Keep the successful intervention fixed.

Reuse the exact completed 4B augmentation materialization: 8,688 trajectories,
40% replaced, independent demanded verdicts and balanced message roles, unchanged
regular instruction and Kimi soft targets. Reuse the same fixed ID, APPS,
honest-control and held-out synthetic ID views. Evaluate the final one-epoch
checkpoint only, with cached original 9B regular results as the control.

Use the previously completed 9B regular run's all-layer nonreentrant checkpointing
and checkpointed shell compilation, BF16 base, FP32 rank-128/alpha-256 adapters,
FlashQLA/SDPA, logical batch 32 and 16,384-token physical packing budget. This
adaptation was already validated at 9B; no new recipe or hyperparameter sweep.
Reuse its startup receipt rather than repeating probes. The startup helper
previously allowed only uncheckpointed receipts; extend it to the recorded
32-layer 9B layout and require matching live nonreentrant checkpointing before
binding kernels. Preserve strict negative results and skipped-check markers.

The complete token audit is reusable because the materialized input hashes and
all three tokenizer files match between the 4B and 9B conditions. Freeze those
identities and the audit receipt checksum; do not claim a new tokenizer pass.
Keep runtime finite/missing-gradient guards and new adapter-specific serving
parity. Collect and verify weights, scores and logs before shutting down.
No publication or additional billable capacity is requested.

See the [protocol](../../experiments/monitor_injection_augmentation_9b/README.md).
