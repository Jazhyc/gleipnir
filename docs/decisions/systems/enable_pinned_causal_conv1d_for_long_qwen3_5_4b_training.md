# Enable pinned causal-conv1d for long Qwen3.5-4B training

Date: 2026-09-03. Status: recorded in the decision log.

- On the Stack 24 two-H100 `gleipnir-improvement` target, source-build pinned
  `causal-conv1d==1.6.2.post1` into an isolated target and require a BF16 GPU
  forward/backward preflight before the mixed-4B model is imported.
- Keep pinned FLA 0.5.2 and causal-conv1d as independent isolated installs. Set
  `CUDA_HOME=/usr/local/cuda` for both so TileLang uses one coherent CUDA 13.0
  compiler/header toolchain rather than mixing packaged CUDA 13.2 and 13.3
  components.
- Adopt the fast path because the matched 8,192-token benchmark improved the
  Qwen-width depthwise convolution by 2.17x and its full gated-delta sublayer by
  1.60x. Do not extrapolate the sublayer ratio to whole-model training; preserve
  the established batch recipe until a matched longest-sequence model preflight
  reports end-to-end throughput and memory.
- Record availability, required status, pinned version, and concrete bound
  causal-convolution module in each training artifact. Fail rather than silently
  returning to Transformers' Torch fallback when the launcher requested the
  extension.
