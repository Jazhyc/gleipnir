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
## Warmed attention bottleneck diagnostic

On 2026-10-05, `profile01` completed a bounded attention-only diagnostic on the
same authorized NC2 B200 and persistent caches. Identical synthetic BF16
operands use causal D256 GQA (16 query / 4 KV heads). Six warmups precede ten
uncaptured repetitions per backend/shape; backend order alternates by shape.
Twenty graph replays separately measure the path with most host dispatch
removed. One warmed forward/backward per condition supplies a CUPTI trace.
No model or optimizer updates occur and existing native validation is reused.
All diagnostic outputs and input gradients are finite.

| Controlled lengths | FA4 wall ms | MXFP8 wall ms | FA4 graph ms | MXFP8 graph ms |
| --- | ---: | ---: | ---: | ---: |
| 301, 280, 250, 230, 201 | 0.819 | 5.181 | 0.101 | 0.274 |
| 4 x 4,096 | 2.404 | 5.207 | 2.294 | 2.483 |
| 14,373, 1,000, 992 | 6.419 | 5.535 | 6.498 | 5.181 |
| 24,521 | 17.926 | 12.903 | 18.274 | 12.683 |

These measure the complete attention call and its input gradients, including
quantization, scale conversion and scratch work. The 24,521 length is an actual
long-singleton length in the measured training window; mixed lengths are
controlled examples, not asserted to be actual training packs. Do not equate
these timings with a complete update or combine profiler times with warmed wall
times as if they were one measurement. Clock/order variation is visible in
graph samples; this short diagnostic is not a repeated throughput campaign.

The MXFP8 path issues 38–39 actual GPU kernels versus FA4's four. Short inputs
are especially sensitive to its Python/FFI launches and allocations: graph
replay cuts MXFP8 short-pack latency from 5.181 to 0.274 ms. Even without most
host dispatch, that shape remains slower than FA4. Eight external quantizer
launches and eleven scale-repack launches persist after removing per-example
dense dispatch. On the balanced pack, quantization and repacking use 0.225 and
0.166 ms of actual GPU kernel time; total MXFP8 kernel time is 2.500 ms, versus
2.380 ms for FA4. GPU conversion overhead erases the native compute advantage
on that shape, with further uncaptured launch overhead.

On the long-singleton trace, actual GPU kernel durations are:

| Operation | BF16 FA4 ms | MXFP8 ms |
| --- | ---: | ---: |
| Forward main kernel | 3.244 | 1.898 |
| dQ main kernel | 4.868 | 4.905 |
| dK/dV main kernel | 8.148 | 4.682 |
| All actual kernels | 16.336 | 12.241 |

The FP8 dQ path is roughly unchanged, while forward and dK/dV improve. Native
backward plus its prologue occupies 9.729 of 12.241 ms (79.5%) of MXFP8 GPU
kernel time. Quantization and repacking contribute 0.327 and 0.254 ms on this
shape. This identifies dQ as a specific remaining kernel target; it does not
prove whether its underlying limit is instruction scheduling, memory traffic,
softmax/online dS conversion or occupancy. That requires instruction/hardware
counter profiling. Do not attribute the long-input limit mainly to repacking.

Complete-update gains also depend on the unchanged 24 GDN layers, all MLPs,
projections, FP32 LoRA work and optimizer. The measured ten-update workload has
73 physical rows, hence 58.4 full-attention calls per update across eight layers;
33 rows are long singletons and seven have maximum length below 4,096. Faster
long attention competes with overhead on other packs. This diagnostic has no
whole-model GPU breakdown, so it does not establish the exact attention share
or rank GDN against MLP costs. For illustration only, reducing a component
occupying 20% of update time by 28% saves 5.6% overall before regressions on other
shapes. The historical complete-update 3.64% difference remains unreplicated.

The [Meta implementation report](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)
describes fused RMSNorm/GEMM-plus-quantization producers and transpose-invariant
square block quantization. Our producer boundary remains BF16, and separate
row/column quantization and scale conversion are explicit. Prioritize reducing
that dispatch/conversion work and investigating native dQ before another
precision-only screen; retain the unchanged numerical failures and FA4 default.

Artifacts are under `results/b200_nvidia_mxfp8_varlen/profile01/`; the timing
receipt SHA256 is
`679a62d8ad3d14dbca19e2c0f36ab9b2918f493de2fe8de71c39125e8ab669fe`.
Preserve its original `operators`/`kernels` fields: the first profiler export
also includes synthetic GPU annotation ranges there, which must not be summed
as actual kernels. `kernel_attribution.json` is the corrected analysis, using
only Chrome `cat=kernel` events and runtime launch correlations to map nested
CPU annotations. It excludes synthetic annotation ranges and binds all eight
traces, the original timing receipt and analysis source by SHA256. Reproduce
without GPU execution:

```bash
python -m experiments.b200_nvidia_mxfp8_varlen.profile_attention \
  --output results/b200_nvidia_mxfp8_varlen/profile01 --analyze-existing
```

Four focused attribution tests cover exclusion of synthetic annotations,
thread separation, duplicate external IDs and coarse external-ID correlation.
Ruff and diff checks pass. The native kernel implementation is unchanged.
