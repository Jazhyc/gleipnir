# Direct variable-length NVIDIA MXFP8 attention

2026-10-05. The user requested adding native variable-length support after the
dense MXFP8 screen averaged 5.29781 seconds/update versus 4.08648 for the matched
historical BF16 FA4 control. This is a separate experimental backend,
`nvidia_mxfp8_varlen`; keep BF16 FA4 as the standard.

The upstream D256 forward kernel already supports packed THD. Its native dQ and
dK/dV kernels accept cumulative sequence offsets, although the shipped backward
adapter hardcodes dense geometry and disables variable-length operation.
The project host now enables those branches with runtime token totals, maximum
length and batch size. Sequence shapes do not enter its compilation key.
The forward adapter declares a bounded envelope of 32 examples and maximum
length 131,072; current Qwen screening inputs remain within their original
context envelope.

Eight whole-row quantization operations replace per-example producers. GPU
CTAs address packed sequence tiles and retain sequence-local 32-element blocks;
they emit both packed forward scales and canonical backward scales. Eleven
whole-row scale conversions retain the pinned native backward's 2-CTA layouts.
The interface uses the collator's shared Q/K cumulative tensor and host maximum
length without copying GPU lengths to CPU. BF16 boundaries, FP32 adapters,
BF16 MLPs and all 24 pinned FlashQLA GDN layers remain unchanged.

Fresh native receipt:
`results/b200_nvidia_mxfp8_varlen/native05/kernel_canary.json`, SHA256
`d1b09cdf6f9bfff141fb048bb61a6c453d669a9e0bc056b605567e3853966d58`.
Ten mixed lengths `[1, 3, 31, 33, 63, 65, 127, 129, 255, 257]` total 964 tokens.
For 16/4 heads, both row and column quantization payloads, canonical scale bytes
and packed scale tiles match the pinned producer. Packed output and all three
gradients match the original dense MXFP8 path exactly. Exact BF16 singleton
identity and its zero Q/K and summed V gradients use explicit whole-row GPU
corrections. Cross-example output effects and gradient leakage are both zero.

Independent FP32 attention gives forward relative L2 4.255%, dQ 6.607%,
dK 6.600% and dV 4.624%. The 2% forward/5% gradient strict gate remains failed;
packing does not cure the earlier MXFP8 approximation. Preserve that result
separately from explicitly authorized bounded timing acceptance.

CUDA graph replay with changed device lengths matches fresh dense outputs and
gradients exactly, including a singleton becoming a two-token sequence.
The first capture attempt failed because prior tests initialized autograd leaves
on the default stream. Initializing fresh leaves on the capture stream resolves
the test failure; see NVIDIA's
[capture-failure guidance](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/capture-failures.html).
Historical failed receipts remain under `native01`, `native03` and `native04`.

Poisoned scratch exposed a real backward-tail issue: dK/dV can consume padded
prologue rows that were not written. Explicitly zero the two FP32 auxiliary
arrays; the final poisoned-buffer test is finite and exactly matches clean
execution. The unused FP32 dQ accumulator slot is never accessed by these
pinned kernels and is omitted from allocation. Forward's token-major LSE is
explicitly transposed to backward's head-major LSE. Scratch initialization,
statistics conversion, quantization, scale conversion and allocation all remain
inside timing. No Compute Sanitizer installation was found on this pod.

The bounded whole-model screen uses fresh eager/compiled packing and memory
gates, the frozen initial FP32 adapter and 20 updates (ten warmup, ten measured).
Controls are checksum-bound historical dense MXFP8 and FA4 trajectories;
the FA4 runtime used its original cuDNN rather than the NVIDIA overlay.
No new quality validation or default promotion follows this screen.
