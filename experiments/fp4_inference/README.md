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
