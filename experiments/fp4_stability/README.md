# FP4 stability diagnostic

Hypothesis: the previous eager/compiled loss disagreement must be separated
from low-precision backward instability. First reproduce forward comparisons
with original BF16 MLP bases and native Four Over Six bases, repeated calls,
and prefixes of the existing compiled decoder policy. No optimizer updates in
the diagnostic stage. Keep the existing absolute 0.01 + 1% loss gate.

Then test BF16 input-gradient GEMMs using the BF16-dequantized exact forward
packed weight. Frozen bases do not need weight gradients; LoRA branches retain
FP32 master adapters. This adopts the dequantized-backward idea from
[The 4-bitter Lesson](https://humansand.ai/blog/nvfp4-rl), not its MoE/RL stack.
Native W4A4 forward GEMMs remain explicit CUTLASS, with MSE 4/6 block selection.
The original fully quantized backward remains an independently recorded control.
The decoded frozen-weight cache costs memory and is not a memory optimization.

An explicit follow-up option normalizes each token row by its FP32 maximum,
converts the normalized input to BF16, and quantizes with a fixed global maximum
of one, then rescales the BF16 GEMM result in FP32 and rounds it to BF16. This
unfused diagnostic removes cross-token scale dependence but is not the blog's
fused kernel or a claim of bit-exact TransformerEngine parity. Native decoded
arithmetic and an outlier-row independence canary must pass before model use.

Use the frozen Qwen3.5-4B revision, training inputs/Kimi targets, selected
320-example cohort and global longest-32 preflight from the B200 experiment.
Retain rank 128, alpha 256, AdamW 5e-5, logical batch 32, physical maximum 8,
16,384 padded-token budget, twelve checkpoints, SDPA, pinned FLA/convolution,
and the established selective compiler policy. Attention storage remains NF4
and compute BF16, as in the control. No generation KV cache or final test access.
Record actual hardware and cache state for every run. Match all comparison
conditions on the same GPU; do not transfer timings across GPU families.

Current target: original B200 Pod `alzfug70g5237b` in US-NC-2 at $6.79/hour,
with network volume `ixbh81vf9c`. Its preserved environment was verified after
resume: Python 3.12.3, Torch 2.11.0+cu130, GPU SM100, 183,359 MiB, zero volatile
uncorrectable ECC errors. Frozen inputs and kernel/model caches remain on the
volume; no data migration was required. Each campaign still verifies its own
input identities and native canaries. The infrastructure record documents the
[reservation handoff and unused stopped Pods](../../docs/infrastructure.md#runpod).

Stop on input drift, backend/kernel failure, OOM, nonfinite values, unchanged
adapters after updates, or failed numerical gates. Diagnostic loss comparisons
may record failures but never authorize training past them. No quality promotion
from short trajectories; separate grouped held-out validation remains required.
Inspect startup every 30–60 seconds. No in-chat scheduler is available, so
monitoring applies during the active turn only.

`row_aot_training.yaml` tests the bounded trajectory with `aot_eager` after
that backend matched eager exactly on the matched-cohort forward diagnostic.
Native FP4, BF16 and NF4 MLP conditions share this backend, initialization seed,
data, checkpointing and adaptive batching. Each first performs the global-longest
update, followed by ten matched updates. This tests optimizer stability while
Inductor numerical consistency remains unresolved; timings do not compare
against the original optimized Inductor recipe.

The first NF4 pass is a finite-training result but has different initial adapter
values despite the same seed. `row_aot_nf4_matched.yaml` repeats that condition
from an explicit standard PEFT initialization artifact whose complete tensor hash
equals the native/BF16 initial hash. It retains the global update and ten-step
stop condition. The runner records initialization file checksums and the screen
fails before GPU preflight if the loaded master hash differs. Same seed alone
does not establish matching initialization across dense/quantized module classes.

Completed outcome: native per-token W4A4 MLP forward plus decoded-weight BF16
backward passes the global-longest update and ten cohort steps with `aot_eager`.
BF16 and corrected NF4 controls also pass, with exact initial adapter, order,
batch and learning-rate matching. Cohort peak allocated memory is 135.2 GiB
native, 116.7 GiB BF16 and 134.8 GiB NF4. Native common training-probe loss falls
1.165299 -> 0.779105. Twenty focused tests pass. This establishes bounded LoRA
training viability; Inductor parity, activation-memory savings, matched speedup
and held-out quality remain unestablished. Full evidence and reproduction limits
are in [the finding](../../docs/findings/fp4_training_stability.md).

## FP4 timing replay

`row_aot_timing.yaml` benchmarks only the stable native FP4 configuration.
Hypothesis: the earlier first-pass times overstate repeated step cost because
of compilation and kernel-cache work. Retain all arithmetic, data, initial
adapter hash, batch policy, twelve checkpoints and AOT backend. There are no
additional precision controls or held-out evaluations in this benchmark.

After the usual native kernel, numerical and global-longest update gates, run
the entire frozen ten-batch cohort once as warm-up and three times as measured
replays in the same process. Restore the exact initial FP32 adapters and create
fresh AdamW state and a fresh ten-step scheduler for every pass. Each pass has
one LR-zero step and nine nonzero-LR updates. Do not select or discard slower
batches. Stop after forty cohort steps or on any existing failure condition.

Synchronize CUDA around each complete logical step, including forward,
backward, finite checks, clipping, optimizer and scheduler. Report every step,
warm-up and measured pass times, token-weighted throughput, batch-specific
means, memory and Dynamo graph counts. Report writing, restoration, model
loading, probes and checkpoint export are outside step timing. Loop wall time
also includes report writing. Compilation graphs must stop growing during
measured passes before treating them as warm. Existing disk caches are reused;
the first pass measures this process's warm-up, not a pristine-cache startup.
Native quantization and per-token scaling remain unfused in this prototype.

```bash
bash experiments/fp4_stability/launch.sh experiments/fp4_stability/row_aot_timing.yaml
```

Completed: the three measured pass means are 15.541/15.897/15.600 seconds per
step, with zero new Dynamo graphs in each. All thirty measured steps average
15.679 s (median 15.440 s), delivering 8,382.6 actual tokens/s on the frozen
cohort. The reused-cache warm-up pass takes 177.174 s of step work. Matching
initialization/workload and source hashes pass the collected analysis; final
adapter hashes differ across replays, so bitwise trajectory identity is not
claimed. See the [finding](../../docs/findings/fp4_training_stability.md#warmed-fp4-timing-benchmark)
for scope, memory, provenance and excluded setup costs.

## FP4 overhead profile

Hypothesis: unfused per-token scaling, surrounding AOT arithmetic and repeated
gradient-check synchronization contribute to the warmed 15.679-second step.
`row_aot_profile.yaml` retains the exact stable FP4 recipe and all preflight
gates, warms the full ten-batch cohort once, then profiles batch five's
forward/backward/finite checks/clipping with CPU and CUDA events. Profiling does
not update parameters or enter timing statistics. It starts from the completed
warm-up adapters and verifies that their full hash is unchanged afterward.
No activation values or private inputs are exported in the trace. Stop after
one profile or any existing numerical/kernel/memory gate failure. Candidate
changes must pass their own arithmetic canaries and the unchanged model gates,
then complete a warmed timing replay on identical data. Require at least 5%
lower step time before calling an intervention a useful throughput improvement.
No other precision condition or held-out quality experiment is launched.

Candidates, frozen before their runs:

- `row_aot_fused_timing.yaml`: fuse each row's FP32 maximum/division/BF16 cast,
  and fuse output FP32 rescaling/BF16 cast. Use round-to-nearest FP32 division
  and prohibit arithmetic reassociation. Require bitwise parity with the
  existing row helpers for zeros, outliers and representative dimensions before
  the native/model canaries. Check missing adapter gradients explicitly and use
  the existing norm-one clip with `error_if_nonfinite=True` to reject all
  nonfinite gradients before updates, avoiding 256 individual host waits. Match
  the original warmed timing's full cohort and three-replay protocol. Defaults
  remain unchanged. This combines two overhead interventions; a total speed
  change does not isolate their individual effects.
- `row_inductor_boundaries_diagnostic.yaml`: retain original unfused arithmetic
  while preserving casts and keeping plain RMSNorm, MLP SiLU and both token
  mixers eager. Compile remaining decoder shells with Inductor. Hypothesis:
  these fences prevent small fused arithmetic differences from being amplified
  by FP4 quantization. This is a forward-only matched-cohort/prefix diagnostic;
  stop without any optimizer work regardless of the gate outcome. A passing
  candidate still requires the global-longest update and warmed replay before
  a performance claim. Preserve failed original Inductor diagnostics.

`row_inductor_fused_timing.yaml` combines the fused row/clip-norm candidate
with the eager boundaries and cast-preserving Inductor policy. Launch only
after the forward-only boundary diagnostic passes; retain the global-longest
update, full ten-batch warm-up and three exact-workload replays. Compare its
measured thirty steps with the stable 15.679-second FP4 baseline, requiring
at least 5% lower mean time before recording a useful performance gain.
Stop on every existing kernel, loss, gradient, memory or adapter-update gate.
This combined intervention does not isolate each optimization.

The separate `precision_cast_diagnostic.yaml` retains the original FP4 scaling
and backward while setting `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1`. The pinned
Torch source documents that default fusion can remove intermediate BF16
downcast/upcast pairs; the option preserves eager rounding boundaries. This is
a compiler hypothesis, not an established cause of the previous failure.
Ten-update campaigns automatically run a separate one-update global-longest-32
stage first for each precision condition, with zero warmup. Any failed stage
stops the campaign before the next stage.
`row_precision_cast_diagnostic.yaml` explicitly selects the matched 320 rows
with diagnostics only. Its ten-batch selection size never authorizes optimizer
updates or queues a global optimizer preflight. This checks input-dependent
compiler consistency after the first matched-cohort gate failure.

Initial implementation validation: 13 focused CPU tests passed using Torch
2.11.0+cpu and other exact locked dependencies in isolated `/tmp` overlays;
local CUDA/native dependency reads were stalling on Lustre. Ruff and shell syntax
checks passed. On the restored B200, original FP4 forward/backward canaries
passed at batch sizes 1/2/4/8. Per-token forward plus dequantized BF16 backward
also passed, with maximum relative L2 0.00235 forward and 0.00167 backward;
the outlier-row independence comparison had zero relative error. Artifacts are
`results/fp4_forward_diagnostic/kernel_canary.json` and
`results/fp4_native_stability/kernel_canary.json`. These are isolated arithmetic
checks; full-model numerical agreement and optimizer behavior are separate gates.

```bash
bash experiments/fp4_stability/launch.sh experiments/fp4_stability/config.yaml
```

## Optimized FP4 timing outcome

The Inductor boundary diagnostic matches eager exactly at all tested prefixes.
`row_inductor_fused_timing.yaml` then passes both model gates, longest-input
backwards and the global update, plus the full warm-up and three matched replays.
Measured pass means are 12.368/12.566/12.503 s: **12.479 s overall**, versus
15.679 s for original FP4 AOT (**20.4% lower step time; 1.256x throughput**).
All measured passes add zero Dynamo graphs. Peak allocated memory increases
135.196 -> 138.638 GiB. The combined intervention does not isolate each gain;
quality, serving parity and cold-start speed remain untested.

For the next bounded FP4 experiment, copy this configuration to a fresh output
path and retain its initialization, native/bitwise kernel canaries, numerical
and gradient gates, partial checkpoints, adaptive batching and full warmed
protocol. This does not change the default NF4 recipe. Full receipts and limits
are in [the finding](../../docs/findings/fp4_training_stability.md#faster-warmed-fp4-training).
The B200 remains running idle; no additional campaign is queued.

## Fused activation packing canary

Hypothesis: fusing the per-token FP32 maximum, BF16 normalization boundary,
MSE 4/6 selection and Blackwell packing removes an intermediate activation
matrix and a kernel launch without changing the stable recipe's arithmetic.
This follows the blog's integration direction, not its full-model MoE/RL
training scope. Its final expert forward passes use FP4, while backward GEMMs
use higher precision and the final approximately 15% of layers remain BF16.
Our LoRA branches and frozen-base BF16 input gradients remain unchanged.

`packing_kernel_canary.py` compares a one-program-per-row candidate with the
current fused normalization followed by the pinned FourOverSix quantizer on
the existing B200. Require exact row maxima, packed FP4 bytes, FP8 scale bytes
including padding, and rescaled CUTLASS outputs for zeros, outlier rows,
irregular row counts and both Qwen MLP input widths. Record warmed timing for
the complete normalization/packing operation. Stop after these isolated cases;
do not load the model or update adapters. A failed bitwise gate or slower
representative large matrices rejects this candidate. Keep the winning
12.479-second training recipe unchanged until a candidate passes and a separate
matched full-model replay establishes its benefit. This kernel retains the
upstream FP32 MSE selector; the blog's faster FP16 selector is a separate change.

The first row candidate fails packed parity at width 9,216. A second canary
prohibiting floating-point fusion still changes four selected block scales in
the 513-row case and is also rejected. Preserve both receipts. The next isolated
`--implementation tiled` probe retains the pinned 16x64 quantization tiles,
computes row maxima separately, and fuses normalization into packing without
materializing the normalized matrix. Its acceptance gates and stop condition
remain the same; isolated timing does not establish training throughput.

The tiled canary passes all seven cases with exact packed operands and output.
`row_inductor_packed_timing.yaml` tests this explicit opt-in against the complete
12.479-second optimized recipe. Retain all initial adapters, sources, native
and model gates, BF16 backward, checkpointing, adaptive batching, compiler
boundaries and gradient checks. Re-run the packing canary before model loading,
then the global-longest update, ten warm-up steps and three measured ten-step
replays. Stop on the existing failure gates or after these bounded passes.
Require at least 5% lower matched mean step time for a useful gain; isolated
packing speed does not establish training throughput or quality. Defaults and
all failed row-packing receipts remain intact. Row maxima still use a separate
kernel; the intervention eliminates the normalized BF16 matrix rather than
claiming the blog's entire fused stack.

Completed: native packed-input forward/BF16 backward, both exact eager/compiled
model gates, longest-input backwards, the global optimizer update and all warmed
replays pass. Measured pass means are 12.892/13.021/12.759 s, **12.891 s overall**:
**3.30% slower** than the preferred 12.479-second recipe. No measured pass adds
graphs. Peak allocated memory remains exactly 138.638 GiB. Faster isolated
packing therefore does not establish a full-step speed or memory benefit.
Keep `row_inductor_fused_timing.yaml` as the preferred bounded FP4 recipe; tiled
packing remains an experimental option. This experiment is complete, the B200
is idle, and no further run is queued. Receipts and limitations are in the
[finding](../../docs/findings/fp4_training_stability.md#tiled-activation-packing-follow-up).

## Sequential FP4 throughput candidates

The user authorized testing the following candidates in order. Baseline is the
completed 12.479-second `row_inductor_fused_timing.yaml` replay; retain FP32
rank-128 LoRA masters, higher-precision attention and decoded-BF16 base backward.

1. `row_inductor_shared_timing.yaml`: share exact normalized/packed activation
   operands between each MLP's gate/up frozen bases. Identity and tensor-version
   checks reject unrelated or mutated inputs; each up consumes the entry once.
   Keep higher-precision LoRA branches untouched. Require exact native output
   and dX parity, checkpoint recomputation checks, unchanged full-model gates,
   the global-longest update and the established thirty measured steps.
2. Test a blog-style FP16 4/6 selector with FP32 error accumulation, first in
   isolated operand/output canaries. This is a numerical-contract change; do
   not assert bitwise agreement with the original strict FP32 selector. Record
   block-choice disagreements and error before the existing model gates.
3. Test fewer launches/intermediates around the native forward path, beginning
   with a profile of the preferred configuration. Require unchanged arithmetic
   or explicitly recorded operand differences, and full training gates.
4. Remove redundant resident original BF16 base weights after reference probes,
   then test the resulting budget with fewer checkpoints or larger physical
   batches. Preserve effective batch, initialization and example order; record
   physical partitions and the changed memory/recomputation policy. Quantizing
   saved LoRA activations is a separate method and is not implicitly enabled.

Run GPU candidates sequentially. Freeze each concrete intervention/config before
its GPU test. Stop a candidate on native, model, gradient, input-identity or OOM
failure and continue the next independent candidate from the last accepted
recipe. Stop each timing campaign after one ten-step warm-up and three measured
replays; require >=5% lower complete-step mean before adopting a useful speedup.
Report regressions and failed gates. No held-out quality promotion or external
serving change is authorized by these short systems experiments.

The shared-activation candidate completes all gates and thirty measured steps
at 12.586694 s/step versus 12.479019 s/step: no useful speedup, with unchanged
138.638038 GiB peak allocated memory. Keep sharing disabled in subsequent
independent candidates. The transient cache contains packed activations/scales
for one unchanged gate/up input only; frozen packed and decoded-backward weight
caches persist across steps.

The concrete selector candidate is `row_inductor_fp16_selector_timing.yaml`.
It retains separate exact row normalization and strict frozen-weight packing,
while packing normalized activations with native `mul.rn.f16x2` candidate
products and an FP32 target/error sum. Arithmetic is inspired by official
TransformerEngine PR3068 (merge `b972fa899eddf69fa7812736d24e479e23a83d3d`);
this independent Triton implementation does not assert bit parity with TE.
Before training, require >=99.9% agreement of logical (unpadded) block scales,
<=0.1% increase in activation squared error, <=0.2% relative native output L2,
finite outputs and deterministic packed operands on all seven frozen cases.
Retain the existing native and full-model gates. Record actual disagreements;
if any gate fails, stop this candidate before model loading. Carry shared gate/up
packing forward only if its preceding complete-step experiment passes selection.

The initial selector at `80178f4` fails the final outlier output gate (0.3665%
relative L2 versus the 0.2% limit) and stops before model loading. The guarded
follow-up at `1ac8d00` restores the strict comparison on near-tied 16x64 tiles,
using a relative error band of `1e-4` and upstream-compatible floating-point
fusion. Its seven isolated operand/output cases pass, with exact outputs on
those cases; full-model warmed timing remains a separate selection gate.
The source commits and full receipts preserve both numerical contracts.

For candidate three, `row_inductor_profile.yaml` first profiles the preferred
arithmetic on warmed batch five without an optimizer update or quality claim.
Then `row_inductor_visible_timing.yaml` replaces Python-disabled frozen bases
with compiler-visible opaque forward/dX operators. Real kernels retain the
existing exact row normalization, native CUTLASS output and BF16 decoded-weight
backward; fake kernels describe only output metadata. Per-runtime CPU tensor
keys avoid integer specialization across projections and any GPU `.item()`.
These process-local runtime keys are rebuilt on model construction and are not
portable exported graph artifacts. Keep eager RMSNorm/SiLU/token-mixer fences
and precision-cast preservation. Require exact native Inductor output/gradient
canaries at three widths and batch 1/2/4/8, unchanged complete-model gates, global
update and the full warm-up/three-replay protocol. Stop at any failed gate or
bounded completion; select only >=5% lower complete-step time.

Official TE row-scaled recipe PR2931 at merge
`c74e5aa37a65eda5c1680562119d466c123ca6ae` uses a separate FP32 output-scale
multiply in its dense path; its cuBLAS entry rejects row-scaled NVFP4 tensors.
A library migration therefore does not itself establish a fused dense epilogue
for this recipe. The opaque-operator test targets compiler boundaries instead.
If it fails or yields no useful gain, a separately gated follow-up may restore
`full_attention_and_linear_shell` compilation while retaining the proven eager
RMSNorm/SiLU fences and BF16 precision casts. That changes execution boundaries,
not attention precision; freeze a forward-only diagnostic before any updates.

For candidate four, `row_inductor_offload_4cp_timing.yaml` offloads the original
frozen BF16 MLP references after the initial native/dense probes, preserving
native packed forward weights and the exact decoded-BF16 backward cache on GPU.
Subsequent dense probes relocate one reference at a time outside timing. The
integrated compile/backward/update gates run with references already on CPU.
Record bytes released and allocated memory before/after offload; never move or
cast the FP32 master adapters. A standalone native canary requires exact
outputs, dX and adapter gradients, unchanged packed/decoded pointers and exact
dense reference outputs before/after offload.

The concrete combined intervention reduces twelve checkpoints to
`[0, 10, 21, 29]`, retaining the 16,384-token physical budget, maximum eight
examples, effective batch 32, initialization and order. If the global-longest
or cohort preflight OOMs, stop that run and test the predeclared eight-layer
fallback `[0, 5, 8, 13, 16, 21, 24, 29]` in its separate configuration/output.
Do not train past an OOM or numerical failure. Once a checkpoint configuration
passes, a separate follow-up may increase only its physical padded-token budget
to 24,576 (oversized singleton exception retained). Record changed partitions
and padding, compare the same actual tokens/order/LR sequence, and require the
same complete gates and thirty measured steps. Adopt only >=5% lower complete
step mean; attribute combined gains to the recorded memory/recomputation policy,
not to the FP4 GEMMs alone. Carry forward only a selected preceding compiler
candidate; otherwise retain the preferred strict eager-native boundaries.

The four-checkpoint global backward exhausted GPU capacity after passing its
arithmetic gates. The eight-checkpoint fallback passed the longest-input
backward. Its conditional larger-budget follow-up is frozen separately in
`row_inductor_offload_8cp_24k_timing.yaml`: retain those eight indices and change
only the physical padded-token budget to 24,576. Launch it only after the
eight-checkpoint 16,384-token timing campaign completes successfully.

The first eight-checkpoint global update passed, but artifact writes exhausted
the persistent volume quota before cohort timing began. Preserve that interrupted
directory and rerun the unchanged training policy with
`row_inductor_offload_8cp_retry_timing.yaml` after clearing rebuildable package
download caches. Storage interruption is not a numerical or GPU-capacity failure.

The sequential campaign is complete. Each passing timing condition uses ten
warm-up steps followed by three measured ten-step replays, with no measured
compilation events. Results against the 12.479019-second FP4 baseline:

| Intervention | Mean seconds/step | Step-time reduction | Selection |
| --- | ---: | ---: | --- |
| Shared gate/up packing | 12.586694 | -0.863% | Keep disabled |
| Guarded FP16 selector | 12.458150 | 0.167% | Opt-in |
| Compiler-visible native projections | 12.137727 | 2.735% | Opt-in |
| CPU reference weights, eight checkpoints, 16,384-token budget | 11.995786 | 3.872% | Opt-in |
| Same eight checkpoints, 24,576-token budget | 13.745459 | -10.149% | Keep smaller budget |

The original FP16 selector failed its output gate, four checkpoints exceeded
capacity, and wider full-attention compilation failed model loss agreement;
those candidates have no accepted training timing. None of the completed
interventions clears the predeclared 5% useful-gain threshold. Retain
`row_inductor_fused_timing.yaml` as the preferred bounded FP4 recipe. Individual
gains do not predict a combined recipe or establish a matched pure-BF16 speed,
held-out quality or serving result. See
`docs/findings/fp4_training_stability.md` for source revisions, gates, memory,
partitions, collected artifact checksums and limitations. No run remains queued;
the B200 is idle and all campaign artifacts are collected and saved persistently.
