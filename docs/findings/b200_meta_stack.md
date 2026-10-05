# Meta attention optimization audit and causal B200 screen

2026-10-05. The user requested trying all optimizations from the
[Meta report](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)
on our existing NC2 B200. This campaign audits the complete technique list,
tests available native controls and project adaptations, and explicitly records
architecture-specific or unavailable paths. It does not claim an exact port of
Meta's end-to-end graph.

## Scope and source evidence

Meta public revision: `2889aefba0d03b3eecf779eed9f4097066ecdc5d`.
NVIDIA public revision: `51d9d06b574222378a3d806009accab098e73705`.
An executed source audit, `results/b200_meta_stack/audit01/audit.json`, archives
both source evidence and the actual constructor guards. D256, GQA four and
causal masking each independently fail Meta's public D128/noncausal MHA guards.
Its pipeline additionally constrains the external MX/broadcast-Q combination.
The public NVIDIA MXFP8 projection validator also rejects `want_rstd=true`:
it exposes an inference epilogue without the training normalization statistics.
These are source checks, not GPU executions of unsupported Meta kernels.

| Technique | Result in this campaign |
| --- | --- |
| Persistent jagged forward scheduler | Already present in NVIDIA D256 forward. Test persistent backward scheduling separately. |
| KV pipeline ordering/unrolling | Native pipeline already specialized. Meta's exact pipeline cannot be substituted across D256/GQA/causal guards. No exact port claimed. |
| Online P conversion/scaling | Test fixed scales 16 and 256 with zero rescale threshold. Native forward already uses fixed identity scales and packed instructions. |
| Compact jagged payload / aligned scales | Already present in the direct variable-length implementation. |
| TMEM scale aliasing | Already present in pinned D256 forward. |
| Square online dS scaling | Test warp-shared maxima adaptation. Separate dQ/dKdV launches cannot reuse one dS payload; this is not Meta's exact square fragment geometry. |
| FP16 global dQ reduction | Inapplicable: native dQ accumulates in TMEM and stores once, with no global partial-gradient accumulator. Test wider final stores instead. |
| Square Q/K/dO scaling | Implemented in the preceding screen and retained here. |
| RMSNorm plus quantization | Implement Qwen head normalization, partial RoPE and native FP8 layout preparation in one producer, with a fused training backward. |
| GEMM plus quantization | Test forward-only BF16/MXFP8 Q projection, FP32 LoRA add, norm/RoPE and quantization epilogues. |
| FP8 returned gradients and producer backward | Fused norm/RoPE backward implemented. Native attention still returns BF16; fused FP8 projection backward remains unavailable and is not implemented. |
| End-to-end optimized module | Integrate the validated producer into Qwen's packed attention, retaining FP32 masters, BF16 gates and output projection. Meta's cross-attention graph is different. |

This audit prevents treating a named optimization as missing when the native
kernel already contains it, or silently changing precision or model semantics
to make an incompatible technique apply.

## Native scheduling and scaling sweep

Each candidate executes in its own process using the existing shared caches.
No new capacity, environment upgrade, upstream source mutation or cold cache.
Execution checks cover singleton semantics, mixed sequence lengths, isolation,
changed-cut CUDA graph replay and poisoned dead storage. Independent FP32
strict parity remains failed, recorded separately from execution acceptance.

| Variant | Short / balanced / skewed / long wall ms | Disposition |
| --- | --- | --- |
| Native baseline, native01 | 3.252 / 3.533 / 5.000 / 12.524 | Complete |
| Persistent dK/dV | 2.506 / 3.334 / 5.193 / 12.528 | Correct; no long-sequence gain |
| Wider dQ stores (128 bits) | 3.801 / 3.764 / 4.479 / 11.829 | Correct; advance with fused producer |
| Persistent dQ / both / both plus warp maxima | No timings | Stalled on native execution; owned process groups stopped, receipts retained |
| Warp-shared dS maxima | No timings | Changed-cut replay/reference dQ relative L2 6.73%; rejected before timing |
| Native baseline, native02 | 2.923 / 2.652 / 5.077 / 12.468 | Complete |
| Forward P scale 16 | 4.328 / 3.675 / 5.256 / 12.724 | Correct; slower |
| Forward P scale 256 | 3.871 / 4.033 / 5.421 / 13.185 | Correct; slower |

Ten warmed samples follow six warmups. Short-pack wall timings are noisy and
include host dispatch. They are diagnostic selection measurements, not complete
training updates. Probability variants scale only the MMA payload and matching
E8M0 metadata, leaving row sums/LSE unscaled. Resetting the rescale threshold
bounds unnormalized P at one to avoid saturation when multiplying by 256.
Warp maxima leave zero measured cross-example leakage but fail the replay
arithmetic gate; describe this as an agreement failure, not proven leakage.

## Fused normalization and rotary producer

The producer consumes raw BF16 Q/K (including strided Q|gate projections),
retains Qwen's BF16 normalization/rotary rounding boundaries, and directly emits
compact FP8 payloads plus consumer scales. Frozen head norm weights and rotary
tables receive no gradients. Native BF16 dQ/dK pass through a fused inverse
partial-RoPE/head-norm backward before the existing projection backward.

`native03/producer128` validates this combined producer and wider dQ stores.
FP8 code agreement is 99.999982% for 16 heads and 100% for four heads; all three
native backward scale layouts match byte-for-byte. Producer backward relative
L2 is 0.00120%/0.00125%. Integrated attention forward matches exactly; Q/K
backward differences are 0.00166%/0.00205%, with V exact. Finite, isolation and
changed-length graph replay checks pass. The producer receipt SHA-256 is
`d6812bee76d29daa2b85c7904dc2918172c7091bb63a6e88185184d11508fb6f`.

Long attention plus producer forward/backward takes 12.225 ms versus 16.941 ms
with eager separate producers in that probe. This 27.84% local reduction is
against separate eager operations; ordinary training already compiles some
surrounding operations. It cannot establish a comparable full-model speedup.

## Matched model contract and measurement

The bounded model screen retains original BF16 frozen weights, 256 FP32 LoRA
master tensors (169,869,312 parameters), the same initial master digest
`a6b1d2e9fd89efff9523150a76035a2e5d27900eaae3c7a4820e3b9277078f11`,
source cohort, soft targets, optimizer, logical batch 32, sequence packing and
FlashQLA GDN layers. Fresh eager/compiled packing and longest-input memory gates
are required for the changed producer. Historical strict failures remain failed;
the user's existing timing-only authorization permits at most 20 guarded updates.

The user explicitly declined a duplicate FA4 control. Compare with the existing
checksum-bound FA4, dense MXFP8 and direct-varlen MXFP8 controls, whose physical
row/token contracts must match exactly. These are historical measurements,
not simultaneous replicated controls. One full-model profile runs on excluded
warmup update eight; updates 11–20 are measured without profiling.

Runtime: Python 3.12.3, Torch 2.11.0+cu130, Transformers 5.14.1, cuDNN
9.26.0.51/backend 92600, Frontend 1.31.0, CuTe DSL 4.8.0, TVM FFI 0.1.11,
FA4 4.0.0b33, FLA/fla-core 0.5.2, Triton 3.7.1, FlashQLA revision
`da06429d54b0f577de0a638f451ac8f0b395e0ac`. Hardware is the already-authorized
US-NC-2 B200 (183,359 MiB), 20.4 CPU quota and 234 GiB RAM. Shared network-volume
cache roots remain `.cache/training/shared/gpu-0/{torchinductor,triton,tilelang,tvm}`,
`.cache/training/shared/{cudnn_frontend,cute_dsl}` and
`.cache/training/fa4_4.0.0b33_cute`. Compiler keys handle new variants; outputs
remain separate. No lifecycle action or future heartbeat was scheduled.

The completed `training01` screen averages **4.51828 seconds/update** over the
last ten updates. It is **10.57% slower than historical BF16 FA4**, **14.74%
slower than direct-varlen MXFP8**, and 14.71% faster than dense MXFP8. It also
regresses from the preceding square-producer screen's 4.15062 seconds/update.
Peak allocated memory is 142.472 GiB. All 20 logical updates complete with the
existing finite-loss and missing/nonfinite-gradient checks enabled. Physical
contracts match all three historical controls exactly; the measured cohort
contains 1,314,331 tokens in 73 physical forwards. No new FA4 control was run.

Fresh whole-model eager/compiled packing gradient relative L2 remains failed
at 17.48%/18.66%; both checks record timing-only acceptance and zero measured
cross-example leakage. The separate adaptive-gradient comparison also fails
strict parity (33.64%) while retaining its explicit selected-recipe acceptance.
Do not interpret the much smaller producer-only backward error as whole-model
quality equivalence. The longest-input memory preflight passes with 32 actual
inputs, maximum length 28,733 and unchanged FP32 masters.

Collected artifact receipts:

| Artifact | SHA-256 |
| --- | --- |
| Training summary | `8a8ab0db0ab8ba4286614eb1019bf6b7a71ff605730b56a4d15b1016b2897843` |
| Training metadata | `e61548731b75844baf5a5521790cd59cf6b89bb9124b1ec129e7af6859819005` |
| Exported FP32 adapter | `e36852506eec1146ce919392b3e0c3cb22bf471061466180db4166d539a412c5` |
| Final master digest | `dcad428b67c7268ab495e913f0cdc10d33d46912c0af2ad28c1bdae31307d1f8` |
| Warmup profile trace | `3b4e64589dc05d9c8284d9282ccde78f7b1cacb07f9c1116bcb0c1369a3df77e` |

Local artifact inspection verifies all 256 exported tensors are finite FP32,
with 169,869,312 elements. Sources, configuration, failed checks, tokenizer,
adapter and runtime logs are collected under the ignored result/log trees.

## Full-model profile and limits of attribution

Update eight contains 34,738 actual CUDA kernel events. Pointwise operations
not identified with a component remain in the other/unclassified group, so the
GDN group is not an exhaustive module accounting. Exclude duplicate GPU
annotations and CPU operations from GPU time accounting. Its summed kernel
time is 4,389.674 ms; the union of kernel intervals is 4,376.732 ms over a
5,048.231 ms first-to-last kernel span. This is one profiled warmup update,
not the average unprofiled wall time or a matched baseline trace.

| Recognizable group | Summed GPU ms | Share of summed kernel time |
| --- | ---: | ---: |
| GEMMs (base/LoRA/MLP/projections mixed) | 1,749.979 | 39.87% |
| GDN and convolution | 894.960 | 20.39% |
| Tensor copies / conversions | 315.183 | 7.18% |
| Native full-attention forward | 72.414 | 1.65% |
| Native dQ and output-dot reduction | 170.087 | 3.87% |
| Native dK/dV | 176.408 | 4.02% |
| Attention operand producer / norm backward | 57.916 | 1.32% |
| Other / unclassified | 952.726 | 21.70% |

The identified full-attention contraction kernels account for only 9.54% of
summed GPU kernel time in this candidate; including its named producer/norm
backward raises that to 10.86%. GEMMs and GDN remain much larger targets. This
helps explain why an attention-only reduction need not give a large complete
training gain. It does **not** establish why this candidate regressed relative
to FA4 or the preceding square path. Graph breaks, conversion/dispatch and
changed compiler boundaries remain hypotheses requiring matched attribution.
Do not label all GEMM time as MLP time or extrapolate these shares to the FA4
baseline. Exact kernel names and analysis source checksum are preserved in
`training01/warmup_profile/kernel_analysis.json`.

## Projection producer pilots

These use actual Qwen hidden size 2,560, 16 Q heads of dimension 256, interleaved
Q|gate weight strides, partial RoPE 64, frozen weights and FP32 rank-128 LoRA
updates. Both legs include adapter matmuls and activation preparation. A
separately quantized MXFP8 weight copy is excluded once, explicitly recorded.
Weights, hidden states and adapter matrices are seeded synthetic tensors; the
model configuration supplies the geometry. This is not a real-checkpoint
projection or adapter-specific parity check. The pilot covers Q only, forward
only: gate/K/V projection and projection
backward are excluded. Its separate producer baseline is eager; no full-model
training improvement follows from these timings.

BF16 `native04/projection_bf16` passes arithmetic/finite gates: FP8 output code
agreement is 99.999869% for lengths [31,33,129] and 99.999970% at 4,096 tokens.
The fused GEMM/LoRA-add/norm/RoPE/quantization path takes 0.347 versus 0.710 ms
for the short packed batch, and **0.582 versus 1.233 ms** at 4,096 tokens.
Its receipt checksum is
`8c30086e1a9b0233beedce9fa98a58e8e938bb6888c15498b7f567daac057b84`.

Initial projection attempts retain setup failures for the unsupported direct
PyTorch split-K setting, then the required cuBLASLt selection. Both were fixed
before the successful BF16 pilot. The first actual MXFP8 compile then rejected
integer zero padding for the FP8 weight load; a floating zero padding fix is
isolated to the projection branch and does not alter the completed training
producer. Failed receipts are retained rather than overwritten.


MXFP8 `native05/projection_mxfp8` then completes both arithmetic/finite gates.
FP8 code agreement with the dequantized-operand GEMM oracle is 99.999493% for
the short pack and 99.999499% at 4,096 tokens. Agreement with the original BF16
projection's output codes is only 47.03%/47.20%. This is discrete code agreement,
not relative L2, gradient parity or a quality metric. Quantization changes the
operands; oracle agreement proves the pilot implements those operands correctly.

Its fused path takes 0.436 versus 0.936 ms for the short pack and **0.570 versus
1.339 ms** at 4,096 tokens. These separate pilot baselines differ in noise; the
0.570 ms MXFP8 and 0.582 ms BF16 fused results do not establish an incremental
FP8 speed advantage. The MXFP8 receipt SHA-256 is
`8649387ee7f88772a0e976b1b023c62784927ab876fa5e30508d00b7ff51d7c6`.
Neither pilot implements projection backward or establishes training quality.

## Decision

The selected combined producer/store recipe fails the prespecified minimum
5% complete-update improvement. **Keep BF16 FA4 as the standard.** Preserve the
faster isolated producers, the original direct-varlen path, and every negative
native/full-model receipt as experimental evidence. Exact Meta pipeline/square
dS reuse and FP8 returned-gradient/projection-backward fusion are not ported;
these require additional native architecture work, not a configuration switch.
The remaining targets suggested by this candidate's profile are mixed GEMMs,
GDN and conversion/dispatch integration. Do not infer a solved training recipe
from synthetic forward-only epilogue timings.

Validation: 78 focused tests pass across the new campaign, prior fused/varlen
receipts and shared packed training. Repository-wide Ruff and scoped diff checks
pass. No ignored artifacts or upstream source copies are committed.
