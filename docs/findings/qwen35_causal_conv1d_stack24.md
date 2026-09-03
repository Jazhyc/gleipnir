# Qwen3.5 causal-conv1d on Lambda Stack 24

## Question

Can the Qwen3.5 training stack use `causal-conv1d` reliably on the long-lived
Lambda Stack 24 H100 target, and is the isolated kernel materially faster than
Transformers' depthwise-convolution fallback?

## Environment

The 2026-09-03 probe used `gleipnir-improvement`, a Lambda
`gpu_2x_h100_sxm5` instance in `us-south-2`:

- Ubuntu 24.04, x86-64, NVIDIA driver 580.126.20;
- two NVIDIA H100 80GB HBM3 GPUs, compute capability 9.0;
- Python 3.12.14, PyTorch 2.11.0+cu130, Transformers 5.14.1;
- pinned `flash-linear-attention==0.5.2`, `fla-core==0.5.2`, and
  `causal-conv1d==1.6.2.post1`.

The causal-convolution package was forced to build from source without changing
the locked project environment. Its cold build took 9 minutes 35 seconds; the
result lives in an isolated `/tmp` target and subsequent installs can reuse the
`uv` cache.

## Compatibility result

Both H100s passed forward/backward comparisons against PyTorch grouped
convolution. FP32 maximum absolute errors were below `2e-5`; BF16 errors were at
the expected reduced-precision scale (at most `0.0625` for outputs/input
gradients and `0.5` for weight gradients). Transformers reported both FLA and
causal-conv1d available, and a real `Qwen3_5GatedDeltaNet` BF16 backward pass
called `causal_conv1d.causal_conv1d_interface.causal_conv1d_fn` exactly once.

The first integrated backward exposed an independent Stack 24 issue: TileLang
selected the environment's CUDA 13.2 `nvcc` while compiling against CUDA 13.3
CCCL headers. Explicit `CUDA_HOME=/usr/local/cuda` made it use the coherent
system CUDA 13.0 toolkit and the Qwen layer passed. The launcher environment now
sets that path when present.

## Matched benchmark

The benchmark used BF16 forward plus backward, batch 1, 8,192 channels (the
Qwen3.5-4B gated-delta convolution width), kernel width 4, and 8,192 tokens.
After three warmups, each figure is the median of ten synchronized iterations
on one H100.

| Scope | causal-conv1d | Torch fallback | Speedup |
|---|---:|---:|---:|
| Depthwise convolution | 1.717 ms | 3.729 ms | 2.172x |
| Complete gated-delta sublayer | 6.974 ms | 11.130 ms | 1.596x |

The benchmark script is `scripts/qwen35_fast_kernel_canary.py`. The sublayer
measurement includes QKV and gate projections, FLA's gated-delta rule, gated
normalization, and output projection. It excludes the decoder MLP and the rest
of the model, so 1.596x is an upper bound on the likely whole-training gain, not
a runtime forecast.

## Decision

Use the pinned extension for the mixed Qwen3.5-4B long-trajectory launcher and
fail closed if either accelerated path is unavailable or unbound. Keep the
established QLoRA batch recipe unchanged until a matched largest-rank,
longest-sequence model preflight records end-to-end throughput and peak memory.
