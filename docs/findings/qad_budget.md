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
source/label/confidence/length strata respectively. Both arms complete their fixed
updates and both 3,012-row ID repeats. Startup retains the experimental-BLAS/SDPA API and unused
visual-module exclusion warnings; no missing/nonfinite gradient is reported.

## Completed budget comparison

| Arm | Training minutes | Macro raw pAUROC@20, two passes | Mean macro pAUROC | Pooled AUROC, two passes |
| --- | ---: | ---: | ---: | ---: |
| 32 updates | 9.07 | 0.857677 / 0.857181 | 0.857429 | 0.943960 / 0.942596 |
| 64 updates | 17.41 | 0.863923 / 0.863598 | 0.863761 | 0.942829 / 0.942642 |
| Cached full QAD, 245 updates | 65.93 | 0.875735 / 0.876433 | 0.876084 | 0.949218 / 0.949311 |

The 32/64 arms lose 1.8655/1.2324 percentage points against full QAD, failing
the predeclared maximum 0.5-point loss. Training is 86.24%/73.60% shorter;
these exclude model loading, master reference, merge, serving startup and ID scoring.
Both improve over the optimized un-QAD mean 0.849031, but recover only part of
the full ranking gain. Neither model or reduced budget is automatically promoted.
Sampling, compressed schedule and structural optimizations differ from the full
reference, so the result does not establish an isolated sample-size effect.

Gloom pAUROC is 0.758029/0.756179 at 32 and 0.777200/0.776568 at 64, versus
0.793433/0.793867 for full QAD. STRIDE is 0.957325/0.958182 at 32 and
0.950646/0.950628 at 64, versus 0.958037/0.959000. More short-arm updates improve
Gloom but reduce STRIDE ranking; pooled AUROC and calibration also do not improve
monotonically. Pooled Brier is 0.095514/0.096836 (32) and 0.098510/0.098527 (64);
ECE is 0.045072/0.046641 and 0.059847/0.059486 respectively.

Serving input throughput is 172,968/173,017 tokens/s (32) and 173,476/172,871 (64).
Repeat score MAE is 0.013057/0.012596, maximum difference 0.298452/0.328244,
correlation 0.996216/0.996073 and threshold flips 39/33 for the two arms.
All three serving canaries pass for each new adapter. The largest canary differences
are 0.143609 (32) and 0.138392 (64), despite passing aggregate MAE/correlation;
this does not establish close parity for every example or across complete ID.

FP32 masters are preserved remotely and locally, with SHA256:

- 32: `96afbfc8b4d80a0073c4e372adca18c36d6334ca805dc5f02348d0162a4809a9`.
- 64: `3ff4d9e0ca46dece15b57c751ca13a0a2180041ed49731a6f4473594dc15603c`.

Independent local checks verify all finite updates and row counts, subset training
orders, teacher/config/source hashes, FP32 tensors/master checksums, paired ID
prompt/source-label identities and exact recalculation of both repeats. Each arm
has `verified_comparison.json` under its result directory. The queue is complete;
the serving process exits and the B200 is idle but still running. No OOD scoring
or extra adaptive training follows these results.

Selection/progress: `results/qad-budget-sweep01/`; arm receipts:
`results/qad-budget32-01/` and `results/qad-budget64-01/`. Logs:
`logs/runpod/qad-budget-sweep01/all.log`. Shared caches and previous quality artifacts
are preserved. No in-chat scheduler is available: detached stage automation does
not provide follow-up agent wakeups after the active turn.
