# Native MXFP8 backward in full QAT

The [experiment](../../experiments/native_mxfp8_qat/README.md) changes attention
backward only relative to completed `native-fp8-qat02`: native FP8 projections,
MLP/GDN optimizations, initial FP32 adapter, BF16 teacher cache, strict-filtered
rows, seed/order and fixed one-epoch optimization are held constant. The forward
emits matching LSE using serving hardware-exp2 arithmetic; native backward
replaces the quantized-QKV FA4 surrogate. This is an approximate-gradient quality
variant, not an exact-gradient claim.

The [isolated probes](native_fp8_qat.md#native-mxfp8-backward-probe) already show
bitwise-identical forward, finite gradients, sequence isolation and faster
attention forward/backward through 8K. Their successful checksum-bound receipt
is reused. Full-model initial-adapter gradients and longest-row memory are newly
checked before updates; gradient deviations are reported without a reproduction
gate. The automatic driver then preserves the FP32 master, replaces the owned
native-FP8 ephemeral merge, checks new-adapter serving parity and runs two frozen
ID passes. Fixed final epoch only, no OOD evaluation or promotion.

The descriptive pAUROC@20 gate permits at most 0.005 additional loss against the
matched native-FP8 mean 0.86199027. Original QAD's 0.87608417 remains a separate
quality reference: passing the new gate would not remove the existing regression.
Fifteen focused CPU controls and Ruff pass. The existing NC2 B200 is verified
idle before launch; driver PID 9034 and training child 9037 completed
`native-mxfp8-qat01`. All 245 updates cover the 7,836-row epoch with finite losses
and gradients; both ID passes complete all 3,012 examples.

Simulator/server alignment reproduces the native-FP8 canary exactly: score MAE
0.00369567 and correlation 0.99987279. Short packed/separate probabilities match
exactly. The paired two-short-row full-model check has identical forward logits,
finite FP32 gradients for every adapter tensor and aggregate gradient relative
L2 4.9869% versus FA4; unclipped norms are 18.92220/18.90943. This limited training
cohort does not establish gradient agreement across lengths or held-out quality.
The longest 29,337-token backward passes with loss 0.79923505, finite gradient
norm 15.16367 and peak allocation 135,476,150,784 bytes (126.17 GiB). Numerical
receipts, training order and all archived source hashes are collected and verified
locally. Startup retains the existing experimental-BLAS and unused visual-module
exclusion warnings; no warning indicates an attention fallback or missing gradient.

## Completed timing and quality

Training takes 3,013.82 seconds (50.23 minutes), versus 3,209.82 seconds
(53.50 minutes) for the matched native-FP8/FA4-surrogate arm: a 6.106% reduction.
This whole-model gain is smaller than the isolated attention speedup. The original
QAD run takes 65.93 minutes, but comparison with that run also changes projection
arithmetic and structural optimizations.

| Model | Macro raw pAUROC@20, two passes | Pooled AUROC, two passes | Training minutes |
| --- | ---: | ---: | ---: |
| Original QAD | 0.875735 / 0.876433 | 0.949218 / 0.949311 | 65.93 |
| Native FP8, FA4 surrogate | 0.862314 / 0.861666 | 0.943940 / 0.944714 | 53.50 |
| Native FP8, native MXFP8 backward | 0.845256 / 0.846050 | 0.936544 / 0.937257 | 50.23 |

Mean macro pAUROC is 0.845653: 1.6337 percentage points below the matched control
and 3.0431 points below original QAD. Both passes confirm the additional regression,
failing the predeclared 0.5-point descriptive gate. Keep the native backward
experimental; no default or model promotion occurs. Gloom pAUROC falls to
0.744910/0.745760 from 0.770636/0.768626; STRIDE falls to 0.945602/0.946340 from
0.953993/0.954707. These are one-seed combined native-FP8/native-MXFP8 results;
native backward with the original projection implementation is deferred by the
user and has not been trained/evaluated.

Pooled Brier is 0.100507/0.099817 and ECE 0.047096/0.045693. Serving input
throughput is 170,863/173,492 tokens/s. Repeat score MAE is 0.012371, maximum
difference 0.369966, correlation 0.996406 and 46 threshold flips, versus 34 for
the matched control. Score resolution remains 94 unique pooled scores per pass.
All three new-adapter serving canaries pass the frozen aggregate gate: MAE
0.015225/0.005982/0.015225 and correlation 0.995180/0.999782/0.995180.
Canaries one and three have a maximum score difference of 0.205785, so aggregate
acceptance does not establish close agreement for every example.

The FP32 master is preserved locally and remotely, with SHA256
`c8a1ea33917c811273227ba7f2a022f528b5d23df11945c6159b8233e34ed289`.
Independent verification checks FP32 tensor types, the master/source archive
hashes, complete finite updates, identical training order, paired ID/prompt/label
identities and exact recalculation of both saved metric sets. The receipt is
`results/native-mxfp8-qat01/verified_comparison.json`. The serving process exits
after scoring, leaving the B200 idle and still running; no baseline server is
automatically restored.

Receipts and archived sources: `results/native-mxfp8-qat01/`; logs:
`logs/runpod/native-mxfp8-qat01/all.log`. Shared compiler/kernel caches and prior
masters/results are preserved. No in-chat scheduler is available; detached stage
automation does not provide agent wakeups after the active turn.
