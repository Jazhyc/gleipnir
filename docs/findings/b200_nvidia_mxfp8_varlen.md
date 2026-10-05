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

## Completed whole-model timing

`training01` completes all 20 finite updates. The final ten include 1,314,331
tokens with exactly the same physical rows, logical indices and token counts
as both recorded controls.

| Attention | Mean measured update | Peak allocated GPU memory |
| --- | ---: | ---: |
| Dense NVIDIA MXFP8, historical | 5.29781 s | 146.85 GiB |
| BF16 FA4, historical | 4.08648 s | 145.24 GiB |
| Direct variable-length NVIDIA MXFP8 | 3.93793 s | 145.10 GiB |

Variable-length support reduces measured update time by 25.67% relative to
dense MXFP8 and 3.64% relative to the historical FA4 control. The complete new
invocation takes 528.58 seconds, including fresh model loading, eager/compiled
diagnostics, memory checks, updates and saving. This uses warm shared compiler
caches; it is not a cold-start measurement. The measured update window totals
39.37934 seconds. Actual packed physical rows contain up to 26 examples, within
the declared 32-example envelope; input partitions are preserved from the
matched controls rather than inferred from nominal profile settings.

The saved adapter is byte-for-byte identical to the dense MXFP8 adapter after
its 20 updates: SHA256
`0aade911f1a97ea803c1dd921e49079221bf7ea4fb75d70310a42cd66888745a`.
All 256 master tensors are FP32. Both runs finish at master-state checksum
`b52c12f171e708681fb223f6f2b277af42070c3cbc36da519dc95702dfad147e`
from the same initial master checksum
`a6b1d2e9fd89efff9523150a76035a2e5d27900eaae3c7a4820e3b9277078f11`.
This supports the correctness of the variable-length port for this trajectory,
while retaining the original approximation relative to BF16.

Fresh whole-model strict and 10% learning parity remain failed: eager adapter
gradient relative L2 is 18.7114%, compiled 16.6558%. Isolation remains zero and
the largest-input preflight passes with unchanged masters before training.
The separate adaptive-gradient comparison is finite but fails at 33.6068%;
its existing selected-recipe acceptance is explicitly recorded. Continue to
distinguish these receipts from finite timing acceptance.

Verification checks both remote/local artifact hashes, all 16 archived
execution-source hashes, complete update counts, identical physical contracts,
master identity and FP32 safetensors headers. Important artifacts and logs are
collected; the B200 is idle and remains running. Receipts:

- `results/b200_nvidia_mxfp8_varlen/training01/summary.json`, SHA256
  `da16c9e8572e0610e9e13da0e7ba71afe56ed063e93ba90a90fd133950108391`.
- `nvidia_mxfp8_varlen/causal_adapter/training_metadata.json` within that run,
  SHA256 `8442b71822b61a445b4f8715c903fc720ab8d9e49b538e359bcdad4ba040f8d9`.
- `artifact_audit.json` and `local_verification.json` within that run preserve
  the final collection and comparison checks.

The 3.64% FA4 time reduction falls below the predeclared 5% threshold and lacks a
fresh FA4 replication. Keep BF16 FA4 as the standard; direct varlen MXFP8 is now
a working experimental option with substantially lower overhead than the dense
prototype. No quality-equivalence claim follows.
