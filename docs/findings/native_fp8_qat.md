# Native FP8 QAT and MXFP8 attention backward

The [experiment](../../experiments/native_fp8_qat/README.md) treats native FP8
projection contractions as a new quality variant rather than claiming the
failed [gradient-reproduction screen](qat_training_optimization.md) passed.
It retains the original strict-filtered population, BF16 monitor targets,
initial FP32 adapter, order, learning-rate schedule and fixed final epoch.
The accepted MLP/GDN structural optimizations are also enabled.

## Native MXFP8 backward probe

The repository already has a packed native MXFP8 dQ/dK/dV implementation with
runtime sequence boundaries. The new wrapper obtains token-major softmax LSE
from the repaired, hardware-exp2 serving forward, then reuses the existing
canonical scale conversion, head-major statistics conversion, initialized
scratch and native backward kernels. It does not use statistics from the
different FA4 forward. Quantized singleton forward behavior is preserved, with
the existing identity-gradient STE correction for singletons.

All tested forward outputs are bitwise identical to the current QAT forward.
All gradients are finite, gradient leakage across examples is zero, and changing
one example has zero effect on other examples' outputs. Relative gradient
differences below refer to the existing quantized-QKV FA4 surrogate, not an exact
derivative of the quantizers or independent FP32 attention.

| Packed lengths | Current attention forward/backward ms | Native MXFP8 ms | dQ / dK / dV relative L2 |
| --- | ---: | ---: | --- |
| 1, 3, 31, 33, 127, 129 | 10.136 | 5.639 | 4.72% / 4.71% / 3.55% |
| 4096, 4096 | 8.422 | 6.006 | 4.09% / 4.06% / 3.71% |
| 8192 | 8.261 | 6.145 | 4.06% / 4.04% / 3.73% |

Three warmup and five timed iterations include conversion, scale repacking,
scratch allocation/initialization and finite-output/gradient checks. These are
isolated attention timings with a single backend order, not complete-model
speedups. The working backend remains experimental: longest-model memory,
whole-model gradients and held-out quality have not been tested for this new
forward/backward pairing. Earlier native-vs-BF16 strict failures remain unchanged.
Receipts live under `results/native-fp8-qat01/mxfp8_backward/` and are collected
locally. NVIDIA documents the [MXFP8 attention operations](https://docs.nvidia.com/cudnn/operations/Attention/);
the measured implementation uses the project's pinned cuDNN 9.26/frontend 1.31.0.

## Native FP8 quality campaign

The initial archived-stage launch failed before model loading because execution
by file path omitted the project root from `PYTHONPATH`. Its failed status,
source archive and logs remain under `native-fp8-qat01`. The environment fix
preserves pinned overlays and explicitly adds the project root, with a focused
regression test. The new `native-fp8-qat02` campaign completed on the authorized
NC2 B200; driver PID 8381 has finished. Ten new CPU controls and Ruff pass.

Fresh simulator alignment passes with score MAE 0.00369567, correlation
0.99987279 and nonzero adapter effect; maximum score difference is 0.02889551.
This is aggregate alignment, not exact per-example parity. Packing isolation
and the 29,337-token longest backward pass also succeed. Peak allocation is
137,556,206,080 bytes (128.11 GiB). All 245 optimizer updates have finite losses
and gradients and cover the full 7,836-row epoch.
The first matched updates 11–20 average 13.23681 seconds versus 15.91791 for the
two completed structural controls, a 16.84% time reduction and 23,394 training
tokens/s. The first 640 source IDs match exactly. This is an early timing window,
not a completed full-epoch or quality result; the comparison receipt is
`initial_timing_comparison.json` under the active campaign output.
Unchanged native FP8 GEMM and FP4 decoder checks are explicitly reused with
checksums and source identities.

The quality arm keeps FA4 surrogate backward to isolate native FP8 projections.
MXFP8 native backward is separately implemented/probed and is not silently
introduced into this frozen campaign. The driver automatically preserves the
FP32 master, rebases adapter keys without changing values, replaces only the
owned ephemeral merged-model slot, checks adapter-specific serving parity and
runs two full frozen ID passes. Both passes complete all 3,012 examples. No new Kimi scoring,
Kimi training, OOD evaluation or model promotion occurs. The completed original
research master and ID predictions remain preserved.

## Completed quality comparison

Training takes 3,209.82 seconds (53.50 minutes), versus 3,955.95 seconds
(65.93 minutes) for the original QAD run, an 18.86% reduction. The full-run
comparison includes the structural optimizations as well as native FP8; the
matched initial timing window above compares against structural controls.

| Model | Macro raw pAUROC@20, two passes | Pooled AUROC, two passes | Training minutes |
| --- | ---: | ---: | ---: |
| Previous QAD | 0.875735 / 0.876433 | 0.949218 / 0.949311 | 65.93 |
| Native FP8 QAT | 0.862314 / 0.861666 | 0.943940 / 0.944714 | 53.50 |

Mean macro pAUROC falls from 0.876084 to 0.861990, a 1.4094-percentage-point
regression. This fails the predeclared maximum 0.5-point quality loss; native FP8
remains experimental. Most of the loss is in Gloom, which falls to
0.770636/0.768626 from 0.793433/0.793867. STRIDE also falls to
0.953993/0.954707 from 0.958037/0.959000. The native model still exceeds the
original optimized monitor's 0.847754–0.850308, but recovers less of the BF16 gap
than the previous QAD model. This one-seed result does not isolate every source
of the quality difference or establish OOD transfer.

Pooled Brier is 0.091802/0.092064 and ECE 0.042059/0.042621, versus the previous
QAD's 0.087798/0.087650 and 0.040775/0.039925. Serving input throughput is
174,626/171,938 tokens/s. Repeat score MAE is 0.012032, maximum difference
0.377374, correlation 0.996432 and 34 threshold flips. The previous QAD also has
34 flips; the native variant does not demonstrate improved repeat stability.

All three new-adapter master/serving canaries pass, with MAE
0.004790/0.004761/0.004761, correlation above 0.99987 and nonzero adapter effect.
These short checks do not override the completed ID regression. The persistent
FP32 master is preserved remotely and locally, with SHA256
`5cb91ecfc78ed613f8796b3ab40ea92b3cc4de74976b48836773514d4fa0603f`;
all 256 tensors are FP32. The lower-BF16 export is unused for merging, which uses
the rebased FP32 values. Numerical receipts, predictions and source archives are
collected under `results/native-fp8-qat02/`.
Independent recalculation exactly matches the saved metrics and verifies paired
row/prompt identities, source archives and the master checksum; the receipt is
`results/native-fp8-qat02/verified_comparison.json`.

The final serving process retires after evaluation. The B200 is idle and remains
running; no baseline server is automatically restored. MXFP8 native backward is
still only kernel-tested and was not used in this full training run.

Logs: `logs/runpod/native-fp8-qat02/all.log`; progress and generated stage sources:
`results/native-fp8-qat02/`. The earlier fixed-queue timing worker is gracefully
retired because it cannot accept this new campaign; persistent compiler/kernel
caches remain shared. There is no in-chat scheduling tool; automatic stage
execution does not promise future agent wakeups.
