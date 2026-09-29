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

The locked model constructor also omits `quant_config` on the tied embedding,
so the constructor-only `bf16_fp32_logits_v3` did not install the intervention.
Use a checked post-load hook before engine warmup; require exactly one ordinary
BF16/FP16 vocabulary head and keep its original weight object and embedding alias.
Actual execution proof is still required for `bf16_fp32_logits_v4` and
`gptq_nvfp4_fp32_logits_v3` before interpreting their measurements.

The post-load control produced fractional margins, but its execution logger
was outside vLLM's configured namespace. Use `vllm.gleipnir.fp32_logits` for
visible dtype proof and read the worker's transferred quantizer config. The
`bf16_fp32_logits_v5` control passed with 32 distinct scores, mean/max drift
0.004185/0.019510 and unchanged AUROC, at 9.901755 s. Three-pass head precision
confirmation also requires visible installation and execution proof.

Verified all-MLP fitted FP4 with FP32 logits failed both canaries (master
mean/max 0.050242/0.178143) and development AUROC loss (0.016529), despite
1.3739x throughput. Actual full-vocabulary output dtype was logged. This negative
result does not promote all-MLP FP4 or replace the passing fitted-down finalist.

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

All six Marlin arithmetic checks passed (relative L2 <=0.001661). Gate/up
speedups at M=1/128/2048 were 1.8224x/1.0636x/0.9165x; down speedups were
1.0878x/0.8765x/0.9077x. This weight-only path is interesting for very small
row counts but did not improve large prefill. Preserve the negative large-M
result and keep native W4A4 for the current serving finalists.

## Frozen broader-split confirmation rule

Three-pass BF16 times were 9.936403/9.950902/9.965627 s; MLP FP8 times were
8.046348/8.051037/8.058522 s (1.2360x median speedup). Both had zero score
and raw-margin variation within their engines. MLP FP8 also exactly matched
its first independent start; BF16 showed between-start max score drift 0.030967.
Use `repeat_analysis.py` to audit identities, all repeat files and stored medians,
then report timings, source-wise drift, metrics and original selection gates.

The fitted FP4-down hybrid's three-pass engine was internally stable but failed
the original master canary on restart (mean 0.022632 > 0.02; correlation
0.987979 < 0.99). Its 1.2808x repeated speedup and passing development metrics
do not override that failure. It is rejected as a confirmed serving candidate.

FlashInfer with FP8 KV storage passed original score gates but took 10.530073 s
(0.9447x baseline). Locked vLLM disabled requested KV-scale calibration for the
recurrent hybrid and used default unit scales. Report that actual behavior;
it is not a calibrated-cache result and did not pass the speed screen.

The next frozen MLP FP8 scheduler screen compares 8,192/eight, 16,384/two and
16,384/eight after the initial 8,192/two condition. Require the original
canaries and development quality bounds, then three-pass timing for a selected
finalist. Freeze `fp8_mlp_b8192_confirm` as the matched repeat of the completed
8,192/two condition. Larger-budget recipes are selected only on development
evidence; full-512 confirmation cannot guide budget or precision choices.

MLP-only FP8 with an 8,192-token prefill budget and two sequences passed the
initial score gates in 7.818977 s (1.2722x baseline), with unchanged macro AUROC,
Brier increase 0.001111 and one threshold flip. Its 2.94% gain over the original
MLP FP8 condition needs matched repeats before choosing the final configuration.

Before broader confirmation, test two additional stability mechanisms. Keep
fitted down weights, FP8 gate/up, original schedule and kernels fixed while
changing only CUDA to independently validated Triton activation packing
(`gptq_down_fp8_triton`). Separately test synchronous offline engine scheduling
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`) for BF16 and MLP FP8, and the locked
vLLM batch-invariance mode for BF16. These are prospective development screens
motivated by the measured BF16 between-start drift; use one initial pass and
repeat only passing finalists. Native FP4's tensor-global activation scale
depends on scheduled token groups, so do not claim batch invariance for it.
The official [vLLM batch-invariance documentation](https://docs.vllm.ai/en/latest/features/batch_invariance/)
is advisory; actual locked Qwen3.5 kernel execution and original gates decide
support. No batch-invariance guarantee is inferred from the environment flag.

The fitted-down hybrid with Triton activation packing passed the initial master
and merged canaries in 7.782019 s (1.2783x), with master mean error 0.011119.
Development AUROC was unchanged and Brier fell 0.005680; maximum individual
drift was 0.215123. Freeze `gptq_down_fp8_triton_confirm` with three passes in a
new engine before interpreting this as a stable candidate or a packer fix.

The synchronous BF16 screen passed parity and exactly matched the earlier
three-pass engine's development scores, but took 10.209538 s (0.9743x initial
baseline). Asynchronous scheduling remained enabled. One matching start does
not establish general reproducibility or a speed benefit.

The matching synchronous MLP FP8 screen passed original canaries and exactly
reproduced earlier FP8 scores in 8.308580 s. This was slower than the default
FP8 path's 8.051037 s repeated median, so retain the default process topology.

Batch-invariance mode failed after loading and before warmup with the locked
backend's explicit `GDN_ATTN` unsupported error. Preserve the failed startup;
this mode provides no numerical or throughput measurement for Qwen3.5 here.

## Frozen fused FP8 activation kernel screen

Hypothesis: combining SiLU/multiply and dynamic per-token FP8 packing can reduce
online down-projection overhead while preserving the stock BF16 intermediate
rounding, E4M3 scales and native CUTLASS GEMM. Use only the existing eight
calibration trajectories, real layer 0/16/31 gate/up weights, and M=128/2048.
Require <=0.005 relative L2 against independently decoded quantized FP32 GEMM
and <=0.005 disagreement from the stock CUDA activation/packing path. Record
all errors before timing; stop on nonfinite output or any arithmetic failure.
Use five warmups, matched three-second heating and one 256-call timing window;
include online activation/packing/allocation/down GEMM. Gate/up projection and
offline weight conversion are excluded identically. Expand to a serving screen
only if the mean M=2048 complete-call speedup exceeds 1.05x. Compare against
the actual compiled vLLM path before attributing a serving improvement to fusion.
The kernel window alone cannot establish monitor-score parity.

The fused activation kernel failed its first stock-path arithmetic gate
(relative L2 0.011620 > 0.005), before timing. Native GEMM versus decoded
quantized inputs passed (0.001655). Preserve this negative and exclude it from
serving. The real earlier MLP FP8 profile already shows an Inductor-fused
SiLU/quantization kernel, so a two-CUDA-kernel microbenchmark would not establish
an improvement over compiled serving. See the finding document for evidence.

The Triton hybrid's fresh three-pass engine failed the master canary again
(mean 0.021178 > 0.02), despite identical within-engine scores and a 7.793463 s
median. Reject the layout as a confirmed candidate; packing alone did not
resolve the observed restart sensitivity.

The 8,192/two MLP FP8 fresh-engine confirmation passed both original canaries:
7.831891/7.847283/7.869111 s, median 7.847283 s (1.2681x repeated BF16).
The fixed development quality screen passed. One low-probability row had a
0.002980 score range, with no threshold instability; do not describe these
larger-budget repeats as exactly invariant. The original 2,048/two FP8 repeats
remain exact in the recorded runs.

## Frozen calibration-selected FP4 precision screen

Hypothesis: protect the eight down projections with largest **calibration-only**
fitted BF16-output relative L2, leaving 24 fitted down projections as native
FP4, all gate/up projections as FP8 and attention/GDN/head as BF16. Use the
existing all-layer capture and fitted artifact, never held-out reconstruction
or monitor labels to select layers. Freeze the full calibration ranking in
`gptq_selective_down8.json`: FP8 down layers 12,14,20,21,23,24,25,27.
Use the original 2,048/two scheduler and Triton activation packing. Prove the
loaded down-weight dtypes layer by layer before compiling. Require unchanged
canaries, development quality bounds and >10% warmed gain, followed by a fresh
three-pass confirmation before any full-split selection. Stop on missing or
wrong precision, nonfinite output or artifact identity failure.

The loaded dtype audit confirmed the selective layout, but the 7.963925 s
diagnostic failed both serving canaries and the development AUROC-loss bound
(0.010331 > 0.01). Reject it. Calibration reconstruction ranking did not provide
an adequate proxy for the monitor's final-score sensitivity.

## Frozen Blackwell shape-attribution follow-up

The user asked why the RTX 4080 INT4 kernel gains exceed some observed Blackwell
whole-model gains. Repeat the same baseline eight-row, sixteen-step GPU profile
with CPU operator shapes enabled. Correlate actual GEMM/GEMV kernels to matrix
shapes through external IDs using the existing `shape_summary.py`; map shapes
to unchanged checkpoint/model projections. Stop attribution on absent GPU
events, invalid correlation or unaccounted GEMM time. Preserve trace/config/input
hashes and report MLP versus attention/GDN projection shares. This is an
instrumented partial window, not a fresh serving benchmark or a compute-versus-
memory counter measurement. Historical INT4 single-kernel gains cannot be
treated as an RTX 4080 full-model result; no native W4A4 vLLM run occurred there.

The eight-sequence/8,192 MLP FP8 condition passed original gates in 7.798921 s
(1.2755x initial baseline), with unchanged AUROC, Brier 0.096856, one threshold
flip and maximum score drift 0.143609. Its small scheduler advantage needs repeats.

The shape-aware Blackwell profile matched all 2,448 GEMM/GEMV calls with zero
unmatched GEMM time. MLP gate/up and down were 28.81% and 14.40% of total sampled
GPU time, totaling 43.20%. A 3x improvement of that fraction alone gives about
1.405x overall, explaining the observed kernel/whole-model gap. This is a
partial-profile illustration, not memory-bandwidth diagnosis. The historical
4080 INT4 speedups were isolated kernel pipelines, with no INT4 vLLM run.

The 16,384/two and 16,384/eight MLP FP8 screens also passed original canaries
and the development quality bounds. Times were 7.835435 and 7.779751 s
(1.2696x and 1.2787x initial BF16). Two sequences gave unchanged macro AUROC,
Brier 0.096726 and two threshold flips; eight sequences gave AUROC 0.925620,
Brier 0.095105 and one flip. These single-pass gains are less than 1% beyond
the repeated 8,192/two median, so retain the repeat-confirmed schedule for
broader confirmation unless another precision screen supplies stronger evidence.

## Final confirmation selection

The GDN-only block-FP8 addition to per-channel MLP FP8 passed the initial
development screen in 7.645032 s (1.3012x baseline). Original master mean/max
error was 0.008595/0.034378, correlation 0.996160; merged mean/max was
0.005178/0.016469, correlation 0.998921. Macro AUROC/pAUROC/Brier were
0.925620/0.764463/0.097983. One development threshold changed and maximum drift
was 0.151355. Preserve it as a promising single-pass development diagnostic;
the final choice was already frozen without seeing these outputs. Freeze a
fresh three-pass `mixed_fp8_gdn_confirm` to inspect timing/restart stability,
with no selection using broader-confirmation scores.

Select one final serving candidate using only the existing 32 development rows,
original canaries, disjoint activation screens and matched three-pass timing
and stability confirmation. Record the choice and complete configuration hashes
before viewing additional score labels. Then run unchanged BF16 and the frozen
candidate on the canonical ordered 512-row subset (5,760,843 prompt tokens),
with the same full prompts, response constraint, decision logprobs and score
runner. Do not tune or select a fallback using this confirmation.

Report all-512 throughput and metrics, and separately compute held-out metrics
after excluding all 32 development rows, all 12 activation-capture rows and the
four parity canaries. Also exclude any row sharing either original or transformed
trajectory hashes with these used rows. Freeze that exclusion manifest before
launch. Preserve source/label counts and report source-wise ranking, Brier,
threshold changes, ties and score drift. This is confirmation within the existing
split, not a final-test promotion or cross-model generalization result. If its
AUROC loss exceeds 0.01 or Brier increase exceeds 0.01, retain and report the
negative result with no deployment recommendation. The one-pass broader timing
does not replace the three-pass development stability check.

At 22:49 UTC, freeze MLP-only per-channel FP8, 8,192 batched tokens and two
sequences as the sole broader-confirmation candidate. It passed the original
canaries on two independent starts and three matched passes; its 1.2681x
repeated median gain is supported more strongly than the larger-budget
single-pass differences. Additional GDN/backend screens remain development
diagnostics and cannot replace this choice using confirmation outcomes.

`confirmation.py` binds the source configuration hashes and analysis source to
the trajectory exclusion manifest, rejects changed inputs/lineage/configuration,
and audits exact ordered prediction identity before computing metrics. The
frozen manifest is `results/fp4_inference/final_confirmation_manifest.json`,
SHA256 `685ba9d162c63c85e508298adba773f5f6ae0706ae8e651bf4b01de78195caf8`.
It contains 464 held-out rows (Gloom negative/positive 161/160, STRIDE 55/88),
after excluding 48 used trajectories. No extra rows share their trajectory hashes.
Full configs are `bf16_full_confirmation.json` (SHA256
`70d297832eb410a193a6ce77e8332ea714fd79731bbeb5905b9f3b69d97b6da6`) and
`fp8_mlp_b8192_full_confirmation.json` (SHA256
`c0e6b57c1941ae5ba4acc7409f439f9e46d0df3a02afc4ba6f1e9c2416a00931`).

The frozen full BF16 baseline completed all 512 rows / 5,760,843 tokens in
171.021511 s (33,684.90 prompt tokens/s), with original canaries passing and
successful worker cleanup. The completed frozen FP8 comparison and preselected
held-out metrics are reported below.

Before the first full FP8 outcomes, freeze two additional independent starts
of each identical serving recipe (`*_full_restart_a/b`), changing only output
paths. Their exclusions/configuration/analysis hashes are bound by
`full_restart_a_manifest.json` and `full_restart_b_manifest.json`. Run them if
the existing allocation permits, without tuning or fallback selection from
their results. Report independent-start timing range/median and score drift
separately from the earlier three passes inside one development engine.

The first frozen full pair completed and passed: BF16 171.021511 s versus FP8
135.557019 s, **1.2616x** warmed speedup. The 464 held-out rows had macro AUROC
loss 0.000207 and Brier change -0.000451, inside the original 0.01 bounds.
Held-out mean/max drift was 0.009575/0.092667, correlation 0.999013 and five
threshold flips. All-512 drift had six flips. Preserve those individual changes;
the result is not exact output equivalence. Detailed source/ranking/calibration
and threshold metrics are in `final_confirmation_report.json` and the finding.

## Frozen wider FP4 diagnostic

Hypothesis: native FP4 in all decoder linear projections can expose a larger
arithmetic speed ceiling than MLP-only FP4. The shape-aware trace attributes
another 26.28% of sampled time to attention/GDN projections. Keep the original
2,048/two schedule, CUDA packing, dynamic tensor-global scales, CUTLASS backend,
embeddings and vocabulary head fixed. Quantize all decoder `LinearBase` layers
using the existing `all` scope, with online weights rather than the MLP-only
prepared artifact. This development diagnostic cannot replace the frozen
broader-confirmation candidate.

Before serving, `shape_canary.py` must bind all four extra N/K shapes to the
observed trace and pass native GEMM versus independently decoded FP32 inputs
at M=1/128/2048 (relative L2 <=0.01; finite output). It uses synthetic BF16
values, establishing arithmetic support only. Record all source/profile hashes,
hardware/software and per-shape errors; stop on any failure. Then one ordered
32-row diagnostic retains unchanged serving/quality gates, including the
explicit finite-failure override and no promotion after failure. No full-split
precision selection follows from this exploration.

All twelve extra-shape arithmetic checks passed, maximum relative L2 0.001973.
Proceed with the single all-decoder-linear development diagnostic; arithmetic
support does not waive the original score or quality gates.

All-decoder-linear FP4 completed in 5.659865 s (1.7576x initial BF16), but failed
both serving gates badly (master mean/max 0.128132/0.370277, correlation 0.198104)
and development AUROC loss (0.016529 >0.01). Its higher diagnostic speed confirms
that accelerating more projections can improve total time; it is rejected for
serving. Preserve the failed gate and do not promote on this speed result.

The additional full-start analysis uses `independent_analysis.py`: it rechecks
each bound exclusion/configuration manifest, exact row/result identities,
unchanged serving configurations and one pass per independent engine. It reports
timing medians/ranges, score/margin ranges and threshold-unstable rows while
retaining every per-pair quality/canary failure. The current focused campaign
test set passed 74 tests; no expensive GPU path was invoked by those tests.
