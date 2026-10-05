# B200 MLP/GEMM optimization

Hypothesis: merging the two frozen gate/up GEMMs and compiling the complete
LoRA MLP, or fusing the GEMMs/LoRA adds/SwiGLU with cuDNN, reduces full MLP
forward-plus-backward time and then complete FA4 training update time.
Keep the original BF16 frozen base, FP32 rank-128/alpha-256 adapters, native
causal variable-length FA4 4.0.0b33, all 24 FlashQLA GDN layers, no model
checkpointing, logical batch 32 and the 16,384-token packing budget.

First compare the actual PEFT module contract with nonzero synthetic adapters
using the cached Qwen3.5-4B configuration. Compare eager, compiled eager,
merged gate/up, compiled merged gate/up and cuDNN LoRA-aware forward fusion.
Include all adapter matmuls, casts and backward operations in step timings;
report forward separately. The merged frozen-weight copy is prepared once,
retained as a nonpersistent buffer and explicitly counted as extra memory.
The original masters, state-dict names and frozen parameters are untouched.
cuDNN fusion must add both adapter updates before SwiGLU and preserve the
BF16 rounding boundaries. Its initial backward uses ordinary BF16 GEMMs and
SiLU backward, rather than claiming a fused native training backward.

Use alternating measurement order, six warmups and ten timed repetitions at
193, 4,096 and 16,384 tokens. Require finite outputs, input and all six adapter
gradients, output relative L2 <= 1%, and each gradient relative L2 <= 1% versus
the actual eager PEFT MLP. Include an independent row-isolation check. Stop a
candidate on compile failure, nonfinite/missing gradients, failed arithmetic,
OOM, or ten-minute timeout, and preserve its receipt. Pilot weights are seeded
synthetic tensors: these are diagnostic timings, not checkpoint quality tests.

The default decoder shells already compile MLP operations. Compiled merged
and cuDNN candidates therefore time against a compiled PEFT MLP in the same
process, with eager PEFT retained as the arithmetic reference. Only a passing
candidate with at least 5% faster full forward/backward against this compiled
control at 4,096 and 16,384 tokens advances to a bounded full-model screen. Freeze the
same 320-row cohort and initial adapter used by the recorded FA4 control;
run at most 20 updates, ten warmup and ten measured. A changed MLP requires
fresh packing/isolation/memory gates, with strict 5% packing results separate
from the user-accepted 10% FA4 gradient ceiling. Preserve original failed
strict receipts and reject missing/nonfinite gradients before updates.
Compare exact physical partitions/tokens against the historical FA4 control
and require >= 5% lower complete-update time to recommend follow-up.
No held-out selection, teacher requests, final-test access or default promotion.

Use the user-authorized NC2 B200 and retained network volume/compiler caches.
Outputs: `results/b200_mlp_gemm/`; logs: `logs/runpod/b200_mlp_gemm/`.
There is no in-chat scheduling tool; inspect startup/progress during the active
turn without promising checks after the turn ends. Keep BF16 FA4 as the default
until results support a change. Native source is the existing isolated NVIDIA
cuDNN Frontend revision `51d9d06b574222378a3d806009accab098e73705`.

## Native FP4 dense-GEMM screen

The next hypothesis is that native NVFP4 forward **and input-gradient** GEMMs
can reduce the dominant frozen-base matrix cost after leaving QLoRA, including
the activation packing/scaling overhead that the previous forward-only FP4
screen did not offset. Probe the merged gate/up and down projections and their
transposed input-gradient projections at actual 4B geometry (2560 / 9216), with
193, 4096 and 16384 tokens. Synthetic seeded weights and activation/gradient
operands are used; these are not whole-MLP or training-update measurements.

Use the retained pinned NVIDIA cuDNN Frontend/FROST overlay and Torch runtime.
Weights use global FP32 scaling plus E4M3 scales shared over 16x16 tiles, with
nearest-even E2M1 packing from the original BF16 tensors in both orientations.
Require exact agreement of the decoded forward/backward weight transpose.
Activations and input-gradient operands use dynamic global scaling and per-row
16-element E4M3 scales. Output is BF16. This bounded timing probe uses RNE for
gradients; it does not claim Transformer Engine's stochastic gradient rounding,
Hadamard transforms or full NVFP4 training recipe. FP32 adapter masters and the
selected BF16 FA4 training recipe are unchanged.

Compare paired ordinary BF16 GEMMs, native FP4 with prepacked operands, and
native FP4 including all dynamic activation reductions, allocation, packing,
scale swizzling, GEMM and output descaling. Pack frozen weights once; record
its startup time and resident packed-pair bytes. Six warmups and ten alternating
synchronized wall-time samples are used. Quantization is lossy: separately
record original-BF16 error and require native-versus-decoded-operand relative
L2 <=1%, finite outputs, and exact decoded weight-transpose consistency.
These checks establish arithmetic implementation, not training quality or the
existing strict 5% packed-gradient acceptance. No objective or tolerance changes.

Stop on arithmetic failure or after the twelve shape/path cases, with a
20-minute process cap. Advance to complete LoRA MLP integration only if at least
one path improves by 5% at both long shapes with packing included. Any later
full-model trial needs fresh precision/packing checks and the historical FA4
control contract. No teacher calls or held-out selection occurs in this screen.

After the first FP4 probe passes all twelve arithmetic checks but fails the
speed threshold, the bounded optimized follow-up replaces the Torch global
amax/inverse sequence with two Triton reductions. Check codes, block scales
and inverse scale against the ordinary reduction before timing. Preserve the
initial failed speed receipt. Also capture matched BF16 and FP4 paths in CUDA
graphs, including the caller-to-static input copy in both replay timings. Check
ordinary and changed-input FP4 replay against uncaptured execution so scales
cannot become stale. Dynamic amax, packing, scale clearing, GEMM and output
descaling remain inside capture. The same arithmetic limits, shapes, warmups,
repetitions and 20-minute stop apply. Graph replay measurements are a bounded
implementation feasibility result; integration with dynamically packed training
is not established, and a graph-only speedup does not automatically authorize
whole-model promotion or bypass full-MLP measurement.

The final packing follow-up replaces tensor-wide activation amax with a per-row
amax, block-scale computation and E2M1 packing in one Triton kernel. Tensor-wide
scaling can make one packed example's quantization depend on another's values;
per-row global inverse scales are applied in the BF16 output-descaling kernel.
Frozen weight scaling remains global and 16x16. Record perturbing the first row
by 31.7x and require exactly zero effect on all other rows before timing each
path. Preserve earlier global-scaling receipts rather than attributing their
speed to an already isolated training implementation. Repeat all twelve native
arithmetic, finite, transpose and changed-input graph checks, with matched
BF16/FP4 graph copies and full conversion included, under the same stop rules.
Full LoRA MLP, autograd/compiler integration and packed model validation remain
subsequent gates; these isolated native kernels do not change the default.

A bounded chunked-row follow-up tests whether separating row amax reduction
from block packing helps the wide 9216/18432-element operands. Keep identical
per-row inverse scales, E4M3 scales and E2M1 codes; require bitwise equality of
all three against fused-row packing before timing every case. Pack independent
128-block chunks rather than carrying the whole padded row through quantization
in one CTA. Record Triton register/spill metadata for both implementations to
check the suspected register-pressure cost instead of attributing slowdown to
launch count alone. Repeat the same twelve cases and graph/isolation gates with
no promotion based on isolated GEMM timing.

## Registered FP4 LoRA MLP integration

Hypothesis: merging frozen gate/up FP4 forwards, using native FP4 base-input
backward GEMMs, and keeping the ordinary compiled LoRA/SiLU operations can
retain the isolated GEMM gains in complete MLP forward/backward. Register both
native operators with fake implementations and explicit autograd; frozen base
weights never receive gradients. Preserve all original BF16 base parameters,
state-dict names, and FP32 adapter masters. Prepare packed forward/transposed
weights lazily after CUDA placement, retain owners to prevent pointer reuse,
and reuse immutable weights/plans across compatible calls. Do not retain an
extra merged BF16 weight copy. Keep all activation/gradient scales per row.

Use the same seeded nonzero adapter fixture, model geometry, 193/4096/16384
rows, six warmups and ten alternating synchronized samples. Measure complete
input and all six adapter gradients against compiled ordinary PEFT, first with
ordinary dispatch and then symmetric whole-MLP forward/backward graph replay.
Include caller input copies in both graph legs and all conversion, allocation,
SwiGLU, adapter GEMMs/casts and backwards in candidate timings. Validate changed
inputs and changed live FP32 adapter masters under replay. Do not cache adapter
copies across updates. Compare native output/gradients to a decoded-operand
FP32-matmul oracle using the identical registered gradient quantization contract,
not to an unquantized derivative. Require <=1% output error, <=2% error for each
of the seven gradients (a multilayer implementation check), finite nonmissing
gradients and exactly zero cross-row output effect. Record original-BF16
quantization errors separately. The existing strict 5% full-model packed-gradient
and loss/isolation gates are unchanged; this 2% local oracle limit is not a
training-quality acceptance or an extension of the FA4-specific 10% ceiling.

Stop the integrated pilot on arithmetic/isolation failure, stale graph replay,
missing gradients, OOM or a 30-minute process cap. Whole-MLP graph improvement
must reach 5% at both long shapes before an ordinary twenty-update FA4 model
screen; positive isolated GEMM timings are insufficient. A full-model trial uses
the exact frozen 320 rows, targets, initial masters and physical contract of the
historical FA4 control, with fresh precision/packing/memory checks. No repeated
FA4 control is needed. Record startup/compiler reuse and every failed receipt;
keep the standard BF16 FA4 profile unchanged pending complete evidence.

The conditional full-model FP4 trial explicitly uses `reduce-overhead` decoder
compilation to attempt the graph dispatch measured locally. This changes
compilation mode as well as MLP arithmetic versus the historical FA4 control;
any total-update difference is a combined recipe result, not an isolated FP4
causal effect. Keep dynamic shapes and the same token/context envelope. Require
the strict 5% fresh packing-gradient acceptance for this new precision, preserving
the FA4-specific historical 10% record independently. A failed packing/memory/
finite-gradient gate stops before optimizer updates; do not widen limits to
obtain a speed result. Archive the native-runtime cache report even on failure.
This session has no in-chat scheduling tool; monitor in the active turn and do
not promise after-turn follow-ups.

After the complete MLP pilot fails the 5% speed threshold, a bounded diagnostic
profiles both matched 16,384-token whole-MLP graphs with ten replays per leg.
Preserve the completed pilot rather than changing its selection rule. Retain
all forward, input/adapter backward and caller-copy work and export raw CUDA
traces plus kernel-name/duration histograms. Distinguish activation reduction,
FP4 packing/scale clearing, output descaling, native dense GEMMs, ordinary
adapter GEMMs and compiled elementwise work. Profiler duration sums describe
instrumented GPU events; they are not replacements for synchronized wall-time
samples, model-update timings or quality evidence. Stop after both traces or a
30-minute cap; do not launch the conditional full-model screen on these data.

Results: `integrate01/integrated` passes native arithmetic, all seven gradients,
row isolation and changed-live-master replay, but saves only 1.56%/2.28% complete
MLP graph time at 4096/16384 tokens. It is not selected for full-model training.
The separate `integrateprofile01/integratedprofile` trace finds conversion takes
1.39850 ms versus 0.70543 ms in native FP4 GEMMs at 16384 tokens. See
[`docs/findings/b200_mlp_gemm.md`](../../docs/findings/b200_mlp_gemm.md) for raw
timings, profiler scope, numerical distinctions and receipt hashes. The BF16
FA4 standard is unchanged; this is an explicit experimental integration.

## Hardware packing follow-up

Hypothesis: Blackwell's native nearest-even E2M1 conversion instruction reduces
packing cost compared with the seven-comparison software encoder, preserving
all quantized operands. The opt-in hardware path also writes padded E4M3 scale
slots inside packing so a separate scale-buffer clearing launch is unnecessary.
Keep identical per-row amax, block scales, weight pairs, native GEMM tiles,
BF16 rounding and explicit input/adapter gradient contract. Do not change
precision, global scale scope, adapter masters or the standard training recipe.

Before measuring complete MLPs, require bitwise codes, scale blobs including
padding, and row inverses against the retained software packer at 193/4096/16384
rows and widths 2560/9216/18432. Include zero rows, signed zero, midpoint ties
and changed-input replay. Record registers/spills and ten alternating pack-graph
samples with copies after six warmups. Then run the existing complete-MLP gates
and timing protocol with three same-process legs: compiled BF16 PEFT, compiled
software-packed FP4, and compiled hardware-packed FP4. Require exactly unchanged
native MLP output/all seven gradients versus software packing; retain the
independent decoded oracle, isolation and live-master checks. Only >=5% faster
complete MLP versus BF16 at both long shapes advances to fresh full-model gates.
Stop on mismatch, missing/nonfinite gradients, replay failure, OOM or 30 minutes.
Preserve failed receipts and do not replace historical wall time with profiles.

A separate twelve-case epilogue probe adapts the existing pinned FROST graph,
keeping its exact tile configuration, with a rowwise FP32 descale multiply.
Declare the original raw GEMM tensor BF16 before multiplication, preserving its
rounding even though that tensor becomes virtual. Reduce global memory traffic
by preparing only one FP32 scale per row, then applying it in the GEMM epilogue.
Use hardware packing in both candidate and reference; require bitwise outputs
against the existing separately-descaled GEMM, finite/nonzero outputs, row
isolation and changed-input replay before interpreting timings. Include all
packing/scaling and symmetric input copies. Six warmups and ten alternating
samples cover both orientations of both projections at all three row counts.
Stop on a failed case or 30 minutes. These isolated timings cannot substitute
for a complete-MLP or model-update screen, or promote a recipe.

After the corrected fresh epilogue graph passes all twelve bitwise cases, test
the combined registered-autograd MLP with four same-process compiled graph
legs: BF16 PEFT, software FP4, hardware-packed FP4, and hardware packing plus
row-descaling epilogues. Preserve the explicit BF16 rounding and quantized
input-gradient contract. Repeat the nine packing gates, exact complete-MLP
output/seven-gradient agreement against software FP4, decoded oracle, row
isolation, live-master replay, six warmups and ten alternating synchronized
samples at all three shapes. Only a complete passing candidate with >=5%
gain over BF16 at both long shapes advances to fresh strict full-model gates;
choose the better measured complete MLP candidate rather than add isolated
path savings. Stop at 30 minutes or a failed gate and preserve prior receipts.

Observed conversion follow-up: native packing preserves all nine operand
fixtures bitwise, and the corrected NVIDIA row-descaling epilogue preserves all
twelve projection fixtures bitwise. Their combined complete-MLP graph reduces
time 18.54%/24.42% versus BF16 at 4096/16384 tokens, including forward and all
input/adapter gradients. Ordinary dispatch remains slower than BF16.
`fusedmlp01` therefore selects the combined opt-in path for `trainconv01`.
The full-model attempt fails fresh eager packing parity (67.90% adapter-gradient
relative L2 and loss drift above the existing bound), despite exact isolation,
and stops before compilation, memory preflight or optimizer updates. No model-
update speedup or training-quality equivalence is established. Preserve the
failed receipts and retain BF16 FA4 as the standard. See the
[finding record](../../docs/findings/b200_mlp_gemm.md) for hashes and diagnostics.

After the user explicitly accepts proceeding despite gradient disagreement,
`training_screen --timing-authority TEXT` permits a separate native-FP4/FA4
timing-only attempt with at most twenty fresh updates. Preserve strict and
learning acceptance as false; record the quoted authority and waived loss/
relative-gradient checks. Isolation, finite/missing-gradient checks, fresh
compiled checks and longest-batch memory preflight remain mandatory. Compare
ten measured updates after ten warmups with the checksum-bound historical FA4
control; require exact initial master and physical partitions/tokens. Record
the changed `reduce-overhead` compile mode as a confound in that comparison.
Stop on isolation/finite/replay failure, OOM or twenty updates. Timing acceptance
requires actual metadata for all 32 native forward/input-gradient MLPs and FP32
masters; an ordinary BF16 FA4 recipe cannot use this waiver. No quality claim,
held-out promotion or default change follows from this timing-only experiment.

The first timing-only attempt reaches decoder graph compilation but fails on
an overwritten graph output before any updates. Its retry marks one CUDA-graph
iteration at each root-model physical forward, preserving decoder-segment
outputs until backward completes. Keep all prior receipts, the same pilot and
timing authority, and fresh compiled/isolation/memory checks. CPU hook tests
cannot establish GPU replay correctness; require the retry to demonstrate it.

The boundary retry advances through compiled no-grad execution but segfaults
in native backward before updates. Preserve that failure and collect logs.
The requested speed result is next measured with `--compile-mode default`,
both conversion optimizations and the same timing-only authority and twenty-
update contract. This matches the historical control's compile mode. Require
all finite/isolation/memory checks, and report the ordinary-dispatch outcome
even if it loses the complete-MLP graph pilot's gains.

`traintiming03` completes twenty finite updates using `default` mode, with the
same initial master and all physical partitions/tokens as the historical FA4
control. The measured half averages 19.42944 seconds/update versus 4.08648,
4.75x slower. Twenty-seven measured token shapes are new after warmup and
conversion kernel compilation continues during measurement; report this as
trajectory wall time, not fully warmed kernel speed. Both strict/learning
packing receipts remain failed and explicitly accepted only for timing.
Peak allocated memory is 148.31 GiB, and all saved adapter tensors are FP32.
The local complete-MLP graph gains have not translated to this ordinary training
recipe. Results/checkpoint/logs are collected and the B200 is idle; retain the
opt-in implementations, negative receipts and BF16 FA4 default.

## Exact-shape warmed timing

User request, 2026-10-05: obtain warmed performance after the default-mode
trajectory still specialized new shapes during measurement. Hypothesis: removing
first-use compilation and native plan construction materially lowers complete
FP4 update time. Use `training_screen --warm-shapes --compile-mode default`
with the same explicit timing-only authority, initial adapter, 320-row cohort,
twenty updates, physical partitions and historical BF16 FA4 control. Reuse all
persistent cache namespaces. Do not alter precision, kernels, optimizer or packing.

Before ordinary training, replay all twenty logical batches including full
forward/backward without optimizer updates. Preserve global CPU/CUDA/Python/NumPy
RNG state and the epoch-seeded loader, verify unchanged FP32 master hashes and
empty optimizer state, and remove warmup records from training metadata. Repeat
the sequence until a replay produces no new native plans, Triton JIT
specializations, Dynamo graphs or Inductor graph-cache misses; require at least
two passes and stop after three. Verify each warmup physical contract against
the historical control. Record warmup costs separately from synchronized updates.
Audit every actual update; claim warmed performance only if updates 11–20 show
zero new plans/specializations/compiler graphs. Keep every sample, the existing
finite/isolation/memory checks and failed strict parity. Stop on missing/nonfinite
gradients, changed masters during warmup, shape-contract mismatch or persistent
compilation. No held-out quality selection or default promotion occurs.

Result: `warmed03` completes both replay passes and twenty actual updates.
The second replay and all updates add zero native plans, Triton specializations,
Dynamo graphs or Inductor graph-cache misses. Measured updates 11–20 average
3.74480 seconds versus historical BF16 FA4's 4.08648 (8.36% less time), passing
the five-percent speed rule. The final adapter is byte-for-byte identical to
`traintiming03`, with all 256 tensors FP32; numerical acceptance remains timing-
only. Warmup step durations total 204.84/73.96 seconds, separate from optimizer
timings; the Trainer runtime includes warmup. All artifacts are collected and
the GPU is idle. Retain failed launch receipts, the experimental path and the
BF16 FA4 default; warmed timing does not establish full-corpus wall-time savings.

## Current warmed full-model profile

Profile the same native FP4/FA4 recipe on fixed actual optimizer updates 11, 15
and 20 after exact-shape replay. Hypothesis: remaining cost is identifiable in
native FP4 contractions, dynamic conversion, ordinary BF16 GEMMs, FlashQLA/GDN,
FA4 and host dispatch. Use `--profile-updates --warm-shapes --compile-mode default`
and the same twenty-update/320-row/master contract; no new BF16 control run.
Reuse the checksum-bound `warmed03` startup diagnostics with original strict
failures and timing-only acceptance preserved. Verify current hardware/software
and native source identity; mark reused checks explicitly instead of repeating
numerical and longest-batch probes. Actual updates retain finite/missing-gradient
checks. Warm every actual batch until the replay adds no plans or specializations.

Capture CPU/CUDA events without shapes/stacks/memory profiling. Include complete
forward/backward, gradient checks/clipping and optimizer work. Export raw traces
and operator/kernel histograms; distinguish summed kernel durations, the union
of device intervals, host gaps and CPU synchronization/launch time. Keep opaque
GEMMs and pointwise work unclassified when the trace cannot identify their model
component. Do not treat CPU waits as additive to GPU time or infer an unprofiled
breakdown from instrumented wall time. This run is diagnostic only: its timings
cannot replace the completed 3.74480-second screen or promote the default.
Stop on nonfinite/missing gradients, changed source/runtime, warmup failure,
OOM or a 30-minute process cap. No held-out evaluation or teacher calls occur.
Monitor during the active turn; there is no in-chat scheduling tool.

Result: `warmedprofile01` completes twenty finite updates, with zero new plans,
specializations or compiler graphs after the second warmup pass. All three
selected CPU/CUDA traces are exported and collected. Pooled kernel shares are
GDN 24.93%, ordinary GEMMs 20.17%, FA4 14.24%, frozen FP4 GEMMs 5.65% and
dynamic FP4 conversion 4.31%. SiLU/fused pointwise work is left without module
attribution; copies and unclassified kernels are reported separately. Traced
updates launch 33,520–46,144 kernels and average 0.71491 seconds without device
events (13.96% of their pooled device span). GDN/projection work and dispatch/
copy overhead are better supported targets than further core FP4 contraction
tuning alone. These are instrumented diagnostic shares, not a decomposition of
the previous 3.74480-second unprofiled mean. The profile loss history and final
FP32 adapter differ from `warmed03`; identical initial masters/batches do not
establish exact trajectory replay. Preserve the original strict failures and
BF16 FA4 default. See the [full findings](../../docs/findings/b200_mlp_gemm.md).

Analyze collected raw traces with:

```bash
python -m experiments.b200_mlp_gemm.analyze_full_profile \
  results/b200_mlp_gemm/warmedprofile01/warmed_profile
```
