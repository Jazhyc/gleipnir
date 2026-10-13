# Short QAD budgets

The user requests 32 and 64 updates to assess reduced QAD overhead. The
[experiment](../../experiments/qad_budget/README.md) freezes nested 1,024/2,048-row
views of the original 7,836-row same-prompt BF16 teacher cache. Both arms reset to
the original completed trained FP32 adapter, use the selected lean QAT arithmetic
and retain logical batch 32, 16K packing, BF16 FA4 surrogate backward and FP32
masters. This is a recovery experiment, not fresh direct-Kimi training.

Sampling preserves population proportions across source, label, teacher confidence
bands and length quartiles through a seeded shuffled, largest-quota-deficit
stream. The 32-arm examples/logical batches are the prefix of 64. Both use peak
LR 1e-5 with independently compressed warmup/linear-decay schedules (one/two
warmup updates), so the comparison changes schedule and exposure. Fixed endpoints
only; no ID tuning, OOD scoring, repeated full control or automatic promotion.
Descriptive success permits 0.005 macro raw normalized pAUROC@20 loss against
cached full QAD mean 0.87608417.

Seventeen focused CPU tests and Ruff pass. Live SSH verifies the existing NC2
B200 idle before launching driver PID 11127 as `qad-budget-sweep01`. The automatic
queue prepares checksum-bound subset/target views, trains 32 updates, checks
new-adapter serving parity and scores two frozen ID passes, then resets and repeats
for 64. Unchanged trained-adapter startup and structural receipts are reused;
per-update finite/missing-gradient checks remain mandatory. Both masters are
preserved and the single owned ephemeral merge is safely replaced between arms.
The 32-update arm completes its first finite update (KL 0.00090630, unclipped
gradient norm 0.103783, 41.27 seconds including initial kernel preparation).
Do not treat that first update as warmed throughput. Frozen nested identities,
resolved config hashes and first-arm source archives are independently verified.
The subsets contain 9,886,430 and 19,743,620 tokens, covering 50/60 populated
source/label/confidence/length strata respectively. Training and quality results
are pending. Startup retains the existing experimental-BLAS/SDPA API and unused
visual-module exclusion warnings; no missing/nonfinite gradient is reported.

Selection/progress: `results/qad-budget-sweep01/`; arm receipts:
`results/qad-budget32-01/` and `results/qad-budget64-01/`. Logs:
`logs/runpod/qad-budget-sweep01/all.log`. Shared caches and previous quality artifacts
are preserved. No in-chat scheduler is available: detached stage automation does
not provide follow-up agent wakeups after the active turn.
