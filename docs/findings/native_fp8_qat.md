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
regression test. The new `native-fp8-qat02` campaign is running on the authorized
NC2 B200 as driver PID 8381. Ten new CPU controls and Ruff pass.

Fresh simulator alignment passes with score MAE 0.00369567, correlation
0.99987279 and nonzero adapter effect; maximum score difference is 0.02889551.
This is aggregate alignment, not exact per-example parity. Packing isolation
and the 29,337-token longest backward pass also succeed. Peak allocation is
137,556,206,080 bytes (128.11 GiB). Actual finite optimizer updates are advancing.
Unchanged native FP8 GEMM and FP4 decoder checks are explicitly reused with
checksums and source identities.

The quality arm keeps FA4 surrogate backward to isolate native FP8 projections.
MXFP8 native backward is separately implemented/probed and is not silently
introduced into this frozen campaign. The driver automatically preserves the
FP32 master, rebases adapter keys without changing values, replaces only the
owned ephemeral merged-model slot, checks adapter-specific serving parity and
runs two full frozen ID passes. Final results are pending. No new Kimi scoring,
Kimi training, OOD evaluation or model promotion occurs. The completed original
research master and ID predictions remain preserved.

Logs: `logs/runpod/native-fp8-qat02/all.log`; progress and generated stage sources:
`results/native-fp8-qat02/`. The earlier fixed-queue timing worker is gracefully
retired because it cannot accept this new campaign; persistent compiler/kernel
caches remain shared. There is no in-chat scheduling tool; automatic stage
execution does not promise future agent wakeups.
