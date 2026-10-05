# NVIDIA MXFP8 causal attention on B200

Date: 2026-10-05. Native forward and backward execute for causal D256 GQA.
Strict native numerical parity fails. A separately authorized learning screen
also fails whole-model adapter-gradient parity at 18.71%, exceeding its 10%
ceiling. Cross-example isolation passes. A subsequently authorized timing-only
run completes 20 updates: MXFP8 averages 5.29781 seconds over the last ten,
versus 4.08648 seconds for the reused, matched BF16 FA4 control (29.64% slower).
Keep BF16 FA4 as the standard.

## Intervention and runtime

Use the existing authorized US-NC-2 B200 (183,359 MiB), Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, unquantized frozen BF16 weights,
FP32 rank-128/alpha-256 master adapters, BF16 MLPs and 24 BF16 FlashQLA GDN
layers. Replace only the eight full-attention layers with NVIDIA's dense MXFP8
forward and backward, called separately per packed example. This preserves
sequence isolation but loses fused variable-length batching. Conversion,
scale-layout repacking and allocation remain inside the measured update.

The isolated overlay uses cuDNN 9.26.0.51, Frontend 1.31.0 built from revision
`51d9d06b574222378a3d806009accab098e73705`, Torch 2.11.0+cu130 and CuTe DSL
4.8.0. Both Torch and Frontend report cuDNN backend 92600. Explicit engine
selection is `sdpa_fwd_prefill_sm100_mxfp8` / `sdpa_bwd_sm100_mxfp8`; there is
no alternative CUDA engine fallback. Payloads and scale factors use NVIDIA's
fused quantizer. Returned activations and gradients are BF16.

Existing shared compiler, FlashQLA, FA4 and network-volume caches are retained.
The Frontend and CuTe DSL caches are also persistent shared directories. No new
capacity or teacher requests were issued. See the
[experiment contract](../../experiments/b200_nvidia_mxfp8/README.md) for the
frozen input/initial adapter checksums, pins and bounded stop conditions.

## Native results

Against independent FP32 causal SDPA, all values and derivatives are finite:

| Sequence length | Forward relative L2 | dQ relative L2 | dK relative L2 | dV relative L2 |
| --- | ---: | ---: | ---: | ---: |
| 3 | 3.11% | 8.47% | 8.65% | 3.56% |
| 31 | 4.02% | 6.77% | 6.73% | 4.48% |
| 33 | 4.03% | 6.78% | 6.88% | 4.45% |
| 127 | 4.29% | 6.57% | 6.64% | 4.78% |
| 129 | 4.32% | 6.47% | 6.59% | 4.77% |
| 257 | 4.55% | 6.56% | 6.60% | 4.84% |

All 24 row/column payload and canonical-scale layout comparisons match NVIDIA's
reference bit-for-bit. Strict 2% forward / 5% gradient limits remain failed.
The user's standing tolerance authorizes a separate learning/timing screen,
bounded at 5% native forward, 10% native gradients and 10% whole-model adapter
gradients. Fresh model isolation, loss, compilation and longest-batch checks
remain required; missing/nonfinite gradients always reject updates. These limits
do not establish training-quality equivalence.

Columnwise V scales depend on future tokens within a 32-token scale block.
Perturbing future V values changes earlier outputs by at most 0.0009765625 in
the native test; perturbations across the block boundary produce exactly zero.
This numerical coupling differs from exact high-precision causal attention and
needs quality evaluation. Calls and scales are sequence-local, so the model
screen separately requires zero cross-example coupling.

cuDNN rejects one-token self-attention backward. The initial native attempt and
first whole-model attempt preserve that rejection, with zero optimizer updates.
The integration now handles this case analytically in BF16: output equals V,
Q/K gradients are zero, and V gradients sum over grouped query heads. This exact
case uses no attention matrix multiplication and is checked against FP32 SDPA;
all longer sequences use the selected MXFP8 engines.

## Whole-model parity outcome (training02)

The fresh eager model probe uses the actual soft monitoring loss and the same
initial master digest as earlier screens:
`a6b1d2e9fd89efff9523150a76035a2e5d27900eaae3c7a4820e3b9277078f11`.

| Measurement | Result |
| --- | ---: |
| Independent BF16 SDPA loss | 0.9492289424 |
| Packed MXFP8 loss | 0.9757985473 |
| Adapter-gradient relative L2 | 18.7114% |
| Strict gradient ceiling | 5% |
| Separately accepted learning ceiling | 10% |
| Optimizer updates | 0 |

Loss drift passes the specified absolute-plus-relative tolerance. All repeat,
prefix perturbation and cross-example input-gradient effects are exactly zero
for lengths [1, 3], [63, 65] and [127, 129]; own-example input gradients are
finite and nonzero. The exact singleton case executes in the full model.
Layers 0–2 match exactly; the first full-attention layer (layer 3) differs by
1.36185% relative L2. The discrepancy reaches 8.23150% at layer 31 and
7.91751% after final normalization. This is consistent with accumulated
low-precision attention differences; it does not identify a kernel defect.

Both strict and learning acceptance remain false. The run stops before compiled
model probes, longest-batch memory preflight and training. The planned FA4
control is therefore unrun. No throughput or training-quality conclusion can
be drawn. The 18.71% value is relative to independent SDPA, not a direct MXFP8
versus FA4 gradient comparison. For context, the earlier matched BF16 FA4 probe
reported 8.28666% against the same reference; see the
[FA4 finding](b200_bf16_fa4.md). Native synthetic gradient errors alone were
insufficient to predict the whole-model discrepancy.

Retain this integration as an opt-in research backend. Investigating real-model
QKV distributions, quantization scales or mixed-precision attention passes would
require a new scoped diagnostic; increasing this screen's ceiling after seeing
the result would not validate the current recipe.

## Authorized timing outcome (training03)

After reviewing the numerical rejection, the user explicitly requested a speed
result regardless. Fresh eager/compiled receipts retain failed strict and
learning parity and separately accept timing execution. Adapter-gradient relative
L2 is 18.7114% eager and 16.6558% compiled. Cross-example effects remain zero;
finite/missing-gradient checks and longest-batch memory preflight pass. The
preflight covers 32 examples and 711,225 tokens without changing master adapters.

| Measurement | NVIDIA MXFP8 | BF16 FA4 (reused control) |
| --- | ---: | ---: |
| Complete optimizer updates | 20 | 20 |
| Warmup / measured updates | 10 / 10 | 10 / 10 |
| Mean measured update time | 5.297811 s | 4.086484 s |
| Total measured update time | 52.978114 s | 40.864844 s |
| Peak allocated GPU memory | 146.851841 GiB | 145.241803 GiB |

The recorded control is
`results/b200_bf16_fa4_accepted/flash_attention_4/causal_adapter/training_metadata.json`,
SHA256 `185fa498f8ec31f07ae58a8213584c7a7f5b41738393ae75d97f91229a993364`.
Model/revision, initial master, seed, optimizer, objectives, BF16 precision,
compiler policy, GDN pins, adaptive policy, input checksums and every physical
batch across all 20 updates match. Each measured window covers 1,314,331 tokens.
Both saved adapters contain 256 FP32 tensors and change from the original master;
all 20 logged gradient norms/losses are finite.

A fresh FA4 repeat was started, then stopped as redundant after the user reminded
us of the prior benchmark and the full contract match was verified. Preserve its
raw termination receipt separately: `summary.json` reports a completed candidate
and a failed control subprocess following SIGTERM; `control_stop.json` records
the intentional stop. The successful historical-control comparison is recorded
in `reused_fa4_comparison.json`, not substituted into that raw receipt. There is
no fresh same-runtime replication: the candidate uses isolated cuDNN 9.26.0.51
while the prior FA4 control used the original runtime. Do not claim this measures
the standalone MXFP8 kernels or a quality-equivalent trajectory.

The dense integration is slower on complete updates despite its lower-precision
arithmetic. It executes attention independently per sequence, while FA4 supports
the packed variable-length batch directly. Each sequence needs six forward and
two backward quantization launches plus eleven backward scale-layout repacks;
allocation and host dispatch remain inside update timing. These are plausible
contributors, not a measured breakdown. NVIDIA describes scale repacks alone as
roughly 1–2% of backward in its own setup; do not assign our entire slowdown to
them. Only eight full-attention layers change; GDN, MLP and adapter work remain
unchanged. Direct packed support is a separate optimization hypothesis.

### Compilation and parallel preparation

The frozen selection manifest records 267 distinct lengths, giving 534 exact
forward/backward plans. A scoped profile places most fresh-plan time in CPU-side
CuTe IR generation. Persistent object-cache reuse works. Sixteen independent
workers prepare those recorded shapes in 208.068 seconds, with 266 hits and 268
misses, no bypasses, invalid artifacts or export failures. Median observed plan
times are 0.114 seconds for hits and 8.232 seconds for misses. These figures
include concurrent CPU activity; the scoped profiler additionally has tracing
overhead. The roughly hour-long serial estimate follows from hundreds of
specializations, not one large compilation.

The pod has a 20.4-CPU quota and 234 GiB host-memory limit. Observed total host
usage during preparation is about 33 GiB. Workers finish before the measured
window and execute no attention or model updates. Recorded selection lengths are
preparation hints; diagnostic and materialized training shapes can add plans.
Cold preparations remain in the excluded first ten update timings. Keep caches
on the shared network volume; no sequence padding/bucketing was introduced.

## Artifact provenance

Ignored artifacts are under `results/b200_nvidia_mxfp8/`; logs are under
`logs/runpod/b200_nvidia_mxfp8/`. The root retains the first native failure,
`attempt02/` the complete supported-length native suite, `training/` the first
model failure, and `training02/` the continuation. Executed-source snapshots
and effective commands are retained per attempt. `training03/` retains the
completed timing candidate, parallel-preparation receipts, stopped control and
the separate historical-control comparison. Native attempt02 receipt SHA256:
`44454758699aacf22439ae252eb099a889097ca4665328dddb80f44b9ca2549c`.
Pinned upstream source archive SHA256:
`d924b07d4c0186dfbeaa7a74d9327c01b6f6a0e144a84cdbab2fb501f4832c83`.
Whole-model packing receipt SHA256:
`a361283faee072e95ce101c153408bebfa021b2ff804781e44c3c5181fd57238`.
Both collected receipt hashes match the B200 originals; all eight executed-source
archives and the initial master identity verify locally. The experiment process
has exited and the B200 remains running with its persistent workspace/caches.

Timing candidate metadata SHA256:
`2047927e4484bdc9481fa158a72f03456c65487b87b4434abd48a9d337c39492`.
Reused-control comparison SHA256:
`ff0c8a76f9cdc6c701ec3c29979798dfa42d1a9e309ee477116248a59773c491`.

The prototype and packed-routing regression suite pass 61 focused tests; the
strengthened native-acceptance checks subsequently pass all 11 prototype tests.
The timing/preparation continuation passes 20 focused prototype tests, including
six new manifest/partition tests; scoped Ruff and staged-diff checks pass.
No downstream evaluation, adapter-serving parity or default promotion has run.
