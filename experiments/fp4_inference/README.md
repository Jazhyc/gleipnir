# Blackwell vLLM inference search

## Frozen protocol (2026-09-29)

Hypothesis: native Blackwell low-precision linear kernels and better prefill
scheduling can improve warmed Gleipnir-4B throughput without unacceptable score
drift or runtime failures. Alternate scheduler/compiler tuning with exploration
of FP8, NVFP4 linear backends, attention, and cache precision. Record failed
conditions and distinguish numerical parity from diagnostic quality measurements.

Use the existing ordered 32-row development slice, 338,780 prompt tokens,
SHA-256 `aadb48556b134150e43946ba39d31512498e2d61b8a88a7a24fd9617b5633f03`.
Keep prompts, token counts, one-token constrained 0/1 scoring, teacher-free
labels, and released adapter/base revisions unchanged. Parent 512-row SHA-256:
`f5800ce52b38184bf3854fbf8e7a91257a3774c2f0a859c598c97e26f5829ebf`.
The development slice guides systems selection; neither it nor kernel
reconstruction error establishes population quality. Calibration, if needed,
must exclude these 32 identities and the original serving canaries.

First run the existing bounded eager base/FP32-master/BF16-merge canary on this
GPU, requiring a nonzero adapter effect and the original merge tolerances.
Then establish the merged BF16 vLLM reference with the historical two-sequence,
2,048-token budget and all original serving gates. No historical RTX 4080 speed
is used as the Blackwell baseline. Each serving process keeps one engine alive
across all canaries and the complete ordered scoring pass. Engine changes need
separate processes; batch compatible kernel measurements in one process.

For each candidate record input/model/config/code hashes, software, resolved
backend, initialization/warmup/whole-process times, warmed tokens/s, GPU
temperature/clocks/memory, raw decision logprobs, paired score/margin drift,
threshold flips, ties, source/macro AUROC, pAUROC@20, Brier and diagnostics.
Initially use one timed pass per condition, following the existing protocol;
confirm selected finalists with matched repeats only if time permits. Require
>10% warmed speed gain for interest. Select a deployable systems candidate only
if serving parity passes and development macro AUROC loses at most 0.01 and
Brier increases at most 0.01 against the new baseline. These predeclared small
development-set bounds are screening criteria, not statistical equivalence.

Quantization candidates may complete a **diagnostic** pass after a finite but
failed numerical canary under this user-authorized search. Preserve the original
limits, failed status and an explicit diagnostic reason; do not relabel failure
or promote that candidate. Stop on nonfinite/missing output, OOM, truncation,
wrong identities, compilation/backend failure or a failed baseline gate.
Record reconstruction checks before using a new FP4 artifact or custom kernel.

Compute: Slurm allocation `32267015`, `gpushort`, `roodborst3`, one RTX PRO
6000 Blackwell Server Edition (97,887 MiB visible, SM120), one CPU, 32 GiB RAM.
Driver 610.57.04; locked Torch 2.11.0, Transformers 5.14.1, vLLM 0.24.0,
FlashInfer 0.6.12. Build concurrency and ordinary CPU thread pools are one.
The existing bounded eager reference sets eight Torch threads; this remains
confined to its untimed four-row check on the one allocated CPU.

Allocation expiry is 2026-09-30 01:39:21 CEST. Stop all campaign GPU workers by
01:29:21 CEST (2026-09-29 23:29:21 UTC), leaving ten minutes. No new allocation
or extension is authorized. Monitoring uses the active agent session; no
scheduling tool is available for agent wakeups after the turn ends. Inspect
startup every 30–60 seconds and each completed condition before choosing the next.

## Run

Initial exploitation conditions increase the prefill budget 2,048 → 4,096 →
8,192 at two sequences, then eight sequences at 8,192, then 16,384 tokens at
eight sequences. This permits attributing budget and concurrency changes.
Initial exploration compares per-channel/per-token FP8 and per-block FP8 at
the baseline schedule, followed by per-channel FP8 at the proposed 8,192/eight
schedule. Each configuration is frozen before launch. Quantization diagnostics
explicitly retain failed canaries; the BF16 reference cannot override its gate.

NVFP4 exploration loads the original BF16 checkpoint through the standard
loader and converts only MLP weights initially. Use E2M1 packed values, one
E4M3 scale per 16 weights and a global FP32 scale `amax/(6*448)`. Activations
compute the same global range dynamically for each invocation, avoiding fixed
range clipping and evaluation-data calibration. Include that reduction and
packing cost in timing. Reject nonfinite scales or sample weight reconstruction
relative L2 >0.25; before serving require native GEMM agreement within 0.01
relative L2 against independently decoded quantized FP32 inputs/weights.
Compare native vLLM CUTLASS with FlashInfer B12X on identical projections.
Do not treat quantized-reference correctness as BF16 numerical parity.

Recreate the existing disjoint twelve-row activation capture with
`python -m experiments.int4_calibration.capture --output
results/fp4_inference/capture`. The capture selects eight calibration and four
held-out rows, excluding iteration32/original canaries, and records all hashes.
This bounded SDPA/Torch-fallback hook pass is a documented exception to serving
evaluation. Run `kernel_canary --backend cutlass` or `--backend b12x`, each with
its own output directory. Use layer-0 gate/up and down calibration activations,
one 256-call timing window, five warmups and three seconds BF16 preconditioning;
compare native output to independent packed-value decoding and FP32 matmul.

Inside the allocation, after activating the locked environment and CUDA 13.2:

```bash
python -m experiments.fp4_inference.run --condition baseline
```

Configurations live in `configs/`. Durable logs are under
`logs/slurm/fp4_inference/`, measurements under `results/fp4_inference/`.
Completed or failed output directories are preserved. Every completed condition
gets a finding here and in `docs/findings/blackwell_inference_search.md`.

## Completed artifact reference

The four-row eager master/merge gate passed: mean/max score drift
0.005537/0.017909, correlation 0.999073 and zero threshold flips; maximum
base-to-master adapter effect 0.421917. The bounded reference used the Torch
gated-delta fallback and SDPA. No serving throughput is established by this
artifact check. Evidence: `results/local_inference/reference.json` and the
reference log. The new BF16 serving baseline has started.

The first serving attempt stopped before model loading on a missing campaign
output parent. Its artifacts are retained under `baseline_startup_failure`;
the launcher creates the parent before retrying. No serving measurement came
from the failed attempt.

## Completed BF16 baseline

The new baseline passed both serving gates and scored all 32 rows in 9.947606 s
(34,056.44 prompt tokens/s). Macro AUROC/pAUROC@20/Brier:
0.921488/0.756198/0.098918, with 30 unique scores. Master-serving mean/max
score error: 0.004477/0.017909, correlation 0.998958, no canary flips.
Cold initialization was 210.770 s; warmup 4.080 s; whole process 247.430 s.
FlashAttention 2 and Triton/FLA GDN were selected. One scoring telemetry sample
showed 58 C, 2,377 MHz, 100% utilization and no thermal slowdown. Full evidence
and qualifications are in the finding document and `results/fp4_inference/baseline/`.

## Completed scheduler screen

4,096/two, 8,192/two and 8,192/eight took 9.751906, 9.545245 and 9.561936 s
respectively: only 1.0201x, 1.0422x and 1.0403x versus the baseline. All serving
canaries passed; macro AUROC/pAUROC were unchanged. The 4,096 condition had one
near-threshold flip; both 8,192 conditions had none. No condition cleared the
>10% interest threshold, and concurrency added no demonstrated gain. Retain
the baseline and prioritize quantization/kernel paths. Full paired diagnostics
and per-condition qualifications are in the finding document.

## Completed native CUTLASS FP4 screen

Native implementation error against independently decoded quantized FP32
references was 0.166% for each real layer-0 projection. Complete online FP4
timing (activation amax/packing/allocation/GEMM) was 0.179417/0.111806 ms versus
BF16 0.531873/0.253100 ms: **2.964x/2.264x**. Offline weight conversion excluded.
BF16 reconstruction error was 11.48%/10.51%, which is quantization error rather
than a kernel mismatch. One 256-call window and uncontrolled 54–64 C thermals;
no full-model quality claim. Evidence: `kernel_cutlass/result.json`.

## Completed full-model MLP FP4 diagnostic

All 64 decoder MLP projections ran native CUTLASS FP4; other paths stayed BF16.
The 32-row pass took 7.113271 s (47,626.47 tokens/s), **1.3985x**. Original
canary parity failed (master mean/max error 0.125850/0.255398 and one flip),
retained under the predeclared diagnostic override. Full-split mean/max drift
was 0.024838/0.185333 with two flips. Macro AUROC/pAUROC/Brier was
0.929752/0.801653/0.096866. This is speed potential with insufficient fidelity;
no promotion. Evidence: `nvfp4_mlp_cutlass/` and the finding document.

Per-channel FP8 took 6.959485 s (**1.4294x**) but also failed the original
canary (master mean 0.027231). Its full-split mean/max drift was
0.022805/0.117002 with one flip. It remains diagnostic-only.

The first selective-FP4 attempt failed on a reused AOT graph with a different
parameter layout. Custom quantizer flags and source SHA-256 now enter the
vLLM compile hash through a preserved resolved configuration. The failed
`nvfp4_mlp_keepends` artifacts remain; its retry is `nvfp4_mlp_keepends_v2`.

The retry completed in 7.325875 s (**1.3579x**), confirming the cache fix.
Keeping layers 0 and 31 BF16 reduced master canary mean error to 0.081046,
but parity still failed. Full-split mean/max drift was 0.026114/0.215126,
with one flip; macro AUROC/pAUROC/Brier: 0.925620/0.780992/0.094867.
This precision allocation does not recover the required fidelity.

## Completed B12X kernel screen

Independent quantized-reference error matched CUTLASS at 0.001659/0.001657.
Gate/up and down complete FP4 calls took 0.169318/0.115939 ms versus BF16
0.531860/0.253152 ms (**3.141x/2.183x**). One window does not establish a
backend winner; the full-model B12X condition is the next comparison.
Evidence: `kernel_b12x/result.json`.

## Block FP8 startup failure and alternate paths

Stock per-block FP8 selected `CutlassFp8BlockScaledMMKernel` but failed during
initial profiling with CUTLASS `Invalid status`, before any canary or timed pass.
Artifacts remain in `fp8_block/`. Test the same precision/schedule with the
explicit Triton linear backend (`fp8_block_triton`) to distinguish the native
CUTLASS path from the quantization recipe; the failing shape is not yet isolated.

Additional exploration freezes separate baseline-schedule conditions for BF16
FlashInfer attention, Triton attention, and FlashInfer with FP8 E4M3 KV cache
(dynamic scale calculation). Compare the actual selected backend, original
canaries, and full-split diagnostics. An MLP-only per-channel FP8 condition
keeps attention/GDN projections BF16 to test whether fidelity loss originates
outside the MLP. All numerical gate failures remain diagnostic-only.

## Completed Triton block FP8 and B12X serving screens

Triton recovered the failed block-FP8 recipe: 8.784008 s (**1.1325x**), with
both original canaries passing. Macro AUROC/pAUROC were unchanged, Brier rose
by 0.003178, and full-split mean/max drift was 0.013666/0.113118 with one flip.
It clears the initial interest and quality screens; matched repeats are now
justified for this finalist. It is not yet a repeatability or equivalence result.

Full-model B12X FP4 took 7.238406 s (**1.3743x**) and failed canary parity.
Full-split mean/max drift was 0.026731/0.185333 with two flips, and macro
AUROC was 0.915289. The one-pass backend comparison does not favor B12X;
neither native FP4 path satisfies the fidelity gate.

## Frozen quantizer calibration screen

Hypothesis: a nearest-even Triton FP4 packer with slightly clipped group-16
ranges can reduce reconstruction error, while power-of-two global scales can
reduce sensitivity to batch composition. Compare dynamic versus rounded-up
power-of-two global scales, and weight/activation group-max fractions
`1.0, 0.95, 0.9, 0.85, 0.8, 0.75`. This is 72 fixed recipes, evaluated on
the six captured projections. Select the single recipe with lowest mean
relative output L2 across eight calibration trajectories. Held-out reconstruction
measurements cannot select the recipe; no monitor scores or evaluation labels
enter this screen. Global scale rounding only increases covered range.

Before selection, require independent nearest-even/sign checks, exact padded
scale swizzle checks, <=0.01 reconstruction disagreement against stock packing
at clip 1, finite output, and native GEMM agreement <=0.01 relative L2 against
decoded FP32 references. Include all online quantization costs when timing a
later serving candidate. Preserve all candidate errors and captured input hashes.
Stop on any validation failure. This screen measures reconstruction, not
end-to-end score fidelity; the unchanged serving gate remains required.

```bash
python -m experiments.fp4_inference.quantizer_screen \
  --output results/fp4_inference/quantizer_screen
```

The completed screen selected unclipped dynamic scaling: calibration mean
relative L2 0.094842, held-out 0.095005. Clipping did not improve reconstruction.
Native implementation error stayed <=0.001675; stock versus alternative packer
weight disagreement stayed <=0.007863. Preserve this negative result.

Next compare the validated unclipped Triton packer with the CUDA packer at the
same real shapes and 256-call protocol (`kernel_canary --packer triton`). A
separate power-of-two activation-scale condition tests batch sensitivity;
rounding increases covered range. Source hashes for every custom helper enter
the compile identity, and custom packing remains explicit in configuration.

Completed Triton packing took 0.182475/0.122285 ms for gate/up and down
(2.915x/2.072x versus BF16), without improving on stock CUDA packing. It stays
an experimental primitive, not a speed promotion.

MLP-only per-channel FP8 completed in 8.049067 s (**1.2359x**) and passed
both original canaries. Macro AUROC/pAUROC stayed unchanged, Brier was
0.096286, full-split mean/max score drift 0.012884/0.089178 and one flip.
It is the faster fidelity-qualified screen candidate so far. Matched repeats
are justified for MLP FP8 and Triton block FP8; retain the BF16 reference.

## Frozen bounded profiling comparison

Capture BF16 and MLP-only FP8 with the same eight every-fourth development
rows, longest-input/original-canary warmup, two delayed steps and at most sixteen
profiled engine steps. Require an early two-kernel CUPTI probe and nonzero CUDA
kernel events in the final trace; CPU durations cannot substitute. This
instrumented partial workload diagnoses kernels and does not estimate serving
throughput or quality. The profile preserves each condition's precision and
scheduling settings. Record kernel duration sums separately from elapsed time.

```bash
python -m experiments.local_inference.profile \
  --config experiments/fp4_inference/configs/fp8_mlp.json \
  --output results/fp4_inference/profile_fp8_mlp --early-cupti
```

## Frozen selective precision and finalist confirmation

Separate the two MLP projections: FP4 gate/up with BF16 down, FP4 down with
BF16 gate/up, and each FP4 projection with per-channel/per-token FP8 in its
sibling. Attention/GDN and kept layers stay BF16. These four fixed conditions
test whether one projection carries most fidelity loss and whether a hybrid
recovers more FP4 speed than the MLP-only FP8 finalist. They retain the original
canary gates and explicit diagnostic override; no precision choice observes
the final test set. Validate selector behavior before launching.

Increase the MLP FP8 prefill budget alone to 8,192 at two sequences. Confirm
BF16, MLP FP8 and Triton block FP8 with three warmed passes per persistent
engine at the baseline schedule, preserving fixed order and prefix cache off.
These justified finalist repeats report median/range and paired score variation;
they do not turn a four-row gate into population equivalence.

The Blackwell profile showed 63.81% of summed kernel durations in linears after
MLP FP8, so also freeze mixed FP8 recipes: fast per-channel MLP FP8 plus Triton
block FP8 in all other decoder projections, then separate attention-only and
GDN-only block precision. Preserve embeddings, head and vision in BF16.
Use stock online quantization arithmetic, explicitly require the per-layer
Triton block kernel, and verify the selected final backend. Source/flags enter
compile identity. These test whether smaller activation groups recover the
full-decoder per-channel FP8 canary while accelerating the remaining linears.

The isolated FP4 projections both failed parity. Down-only took 9.047070 s
(1.0995x), with master canary mean/max 0.046753/0.124353 and three full-split
flips. Gate/up-only took 7.852298 s (1.2668x), with canary mean/max
0.057448/0.117002 and two full-split flips. Gate/up provides more speed gain;
neither projection is fidelity-qualified in this basic online FP4 recipe.

## Frozen Hessian-feedback FP4 screen

The basic FP4 precision layouts missed score fidelity. Test fixed-order
GPTQ-style weight error feedback using only the eight disjoint calibration
trajectories. Use covariance `X.T @ X / n`, diagonal damping 0.01 times mean
diagonal, inverse-Hessian upper Cholesky, 128-column feedback blocks, native
16-value weight groups with E4M3 scales, and the original global weight scale.
No activation ordering, hyperparameter sweep, teacher targets, or label objective.
Keep ordinary dynamic FP4 activations. Require an independent small NumPy
feedback reference with exact packed codes/scales and native GEMM agreement
<=0.01 relative L2 against decoded FP32 inputs/weights.

First screen all six existing captured projections. Expand to all MLP layers
only if mean calibration W4A4 output relative L2 improves by at least 10% and
held-out mean error does not exceed stock by more than 2%. This is a fixed
numerical gate, not model promotion. Stop on Cholesky/nonfinite/packing failures;
preserve partial results. A later full-model artifact needs source/calibration
checksums, independent reconstruction checks and unchanged serving gates.

```bash
python -m experiments.fp4_inference.gptq_screen \
  --output results/fp4_inference/gptq_screen
```

The six-projection screen passed independent exact packing and native arithmetic
checks. Calibration mean error improved 27.2% (0.094842 → 0.069032); held-out
mean improved 4.9% (0.095005 → 0.090320). This clears the fixed expansion rule.
The same twelve trajectories may now be captured at all 32 MLP layers; eight
calibration rows fit every projection with the already frozen recipe. Four
held-out rows only audit reconstruction. Record each packed artifact checksum,
all 64 projections, source-weight reconstruction <=0.25 and native agreement
<=0.01 before marking the export complete. Preserve the original BF16 weights.

```bash
python -m experiments.int4_calibration.capture --layers all \
  --output results/fp4_inference/capture_all
python -m experiments.fp4_inference.gptq_export \
  --output results/fp4_inference/gptq_all
```

The complete 64-projection export is checksum-bound to the original BF16 merge
and disjoint capture. Serving requires the frozen manifest hash, each packed
file hash, exact original weight-byte identity, expected shapes/dtypes and finite
scales, followed by the original reconstruction and score gates. Compare
`gptq_nvfp4_triton` (the same activation packer as the numerical screen) with
`gptq_nvfp4_cuda` (stock CUDA activation packing), keeping prepared weights
identical. Native CUTLASS GEMM stays fixed. Frozen prepared-weight hybrid
conditions retain FP8 in the sibling projection. Artifact/source hashes enter
runtime and compiler identity; no fitted or original weights are committed.

## Frozen logits-precision stability screen

BF16 logits have coarse margins and tied scores. Test retaining FP32 output
from the full-vocabulary head matmul while preserving BF16/FP16 head weights
and inputs. Explicitly disable reduced-precision BF16/FP16 intermediate reductions
for this matched pair. No vocabulary selection, prompt/sampling change, score
rescaling, or canary-limit change. First require finite FP32 output and <=0.001
relative L2 against independent FP32 matmul on real captured inputs. Then compare
an all-BF16/head-FP32 control with the frozen fitted-FP4/head-FP32 condition
(Triton activation packing). Keep both original eager references and the old
BF16 serving baseline for diagnostics. Record ties and raw margins as well as
latency and quality. The custom registry's `scope=none` explicitly means no
FP4 decoder projections in the control; it is not an FP4 speed measurement.

Qwen3.5 ties the head to its vocabulary embedding. The initial selector for
untied heads did not install this intervention in `bf16_fp32_logits_v2` or
`gptq_nvfp4_fp32_logits`; retain both as mechanism failures. Corrected conditions
`bf16_fp32_logits_v3` and `gptq_nvfp4_fp32_logits_v2` cover tied embeddings,
preserve ordinary lookup, and require an actual FP32 projection execution log.
The direct GPU method canary already passed with relative L2 below 0.000004.

Use `cluster/slurm/fp4_inference_step.sh` in a detached `srun` step within the
existing allocation so tool-session refreshes do not interrupt serving. The
campaign runner reserves 20 seconds for termination before its fixed stop time;
bounded module calls use the same reserve. Inspect live steps, logs and GPU
telemetry in the active agent turn. No recurring agent wakeup is available.

The fitted FP4-down/FP8-gate hybrid passed both original canaries and the
development screen at **1.2948x**, with zero full-split flips and max score drift
0.061457. Master canary mean 0.019111 is close to 0.02; repeat confirmation is
required. Macro AUROC/Brier met their bounds; pAUROC declined 0.010331.
Freeze a three-pass baseline-schedule confirmation and an 8,192-token/two-sequence
scheduling condition for this new finalist. Keep all quality limits unchanged.

The reverse fitted hybrid (FP4 gate/up, FP8 down) was faster at 1.3528x but
failed merged-canary correlation (0.989657 < 0.99); it remains diagnostic only.

## Frozen Marlin FP4 weight-only kernel screen

Compare native Marlin W4A16 (BF16 activations, fused FP4 weight dequantization,
FP32 reduction) with BF16 at M=1,128,2048 for both real layer-0 MLP shapes.
Reuse the eight checksum-verified calibration captures and original weights;
offline weight packing/repacking is excluded, online padding/allocation/GEMM
included. Require <=0.005 relative L2 against independently decoded weight-only
FP32 references and finite output before timing. Use one 256-call window per
condition, five warmups and three-second BF16 heating. Record thermals/clocks.
This explores a different low-precision path and row-count tradeoff; these
microbenchmarks do not establish full-split serving performance or quality.
