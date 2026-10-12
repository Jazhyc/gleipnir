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
idle before launch; driver PID 9034 and training child 9037 start
`native-mxfp8-qat01`. Three finite updates are collected after fresh startup
passes. Complete timing and held-out results remain pending.

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

Receipts and archived sources: `results/native-mxfp8-qat01/`; logs:
`logs/runpod/native-mxfp8-qat01/all.log`. Shared compiler/kernel caches and prior
masters/results are preserved. No in-chat scheduler is available; detached stage
automation does not provide agent wakeups after the active turn.
