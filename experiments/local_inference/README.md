# Local merged Gleipnir 4B baseline

## Per-channel FP8 follow-up

Hypothesis: replacing per-tensor weight scales with per-output-channel scales
reduces FP8 score drift enough to pass the existing canaries. Installed vLLM's
`fp8_per_channel` uses `Fp8PtpcOnlineLinearMethod`, with per-channel weights and
dynamic per-token activations. The prior CUTLASS per-tensor path already used
per-token activations, so this isolates weight scaling within the supported
linear paths. No layer exclusions, calibration data, checkpoint edits, cache
precision changes, or tolerance changes.

Run `python -m experiments.local_inference.run --config
experiments/local_inference/fp8_channel.json`. Retain the frozen 32-row selection,
one timed pass, and all other baseline settings. Compare to the saved BF16
baseline without repeats. Require the original four-row master/merged canary
limits (mean <=0.02, max <=0.10, correlation >=0.99), then the longest-input
canary before timing. Stop on gate failure, nonfinite/missing output, OOM,
truncation, or identity mismatch. Preserve failed per-tensor artifacts. Report
paired drift, threshold flips, metrics, startup, throughput, memory, and thermals;
>10% speedup merits investigation, not automatic promotion. No population-level
quality claim from these development rows. Monitor during this active turn only.

### Per-channel result: improved drift, original gate still fails

CUTLASS FP8 (`Fp8PtpcOnlineLinearMethod`) loaded successfully. Four-row canary
results, with no 0.5-threshold flips in any comparison:

| Reference | Mean absolute score error | Max error | Correlation |
| --- | ---: | ---: | ---: |
| FP32 master (eager) | 0.024373 | 0.060283 | 0.999648 |
| BF16 merged (eager) | 0.028851 | 0.060283 | 0.997422 |
| Existing BF16 vLLM canaries | 0.016751 | 0.031552 | 0.998485 |

Only the mean-error criterion fails against the original eager references.
Against BF16 vLLM, all three numerical limits would pass, but that was not the
predeclared gate and was not substituted after seeing results. Relative to the
per-tensor FP8 candidate, serving-to-serving mean/max drift improved from
0.057170/0.224908 to 0.016751/0.031552. The sensitive STRIDE score is now 0.377541,
versus 0.407333 BF16 and 0.182426 per-tensor FP8. This is promising canary evidence,
not a full-workload quality result; all four serving score changes were downward.

Stopped before the longest-input canary and 32-row pass. No throughput result or
promotion. Changing the gate/reference to permit a diagnostic timing pass needs
an explicit decision; retain all existing failures. Model allocation stayed
5.0 GiB, KV capacity 6.31 GiB / 194,125 tokens. Initialization took 123.42 seconds,
including 40.68 seconds compilation and 58.93 seconds initial profiling/warmup;
the failed subprocess took 144.96 seconds. GPU worker and telemetry exited.

Evidence: `results/local_inference/fp8_channel/{serving_parity.json,
baseline_canary_comparison.json,initialization.json,process_timing.json}`,
`fp8_channel_status.json` (failed), and
`logs/local/local_inference/20260923T215245Z-benchmark.log`. Child status remains
at its last phase, `loading`, as in the earlier failed screen.

## FP8 online-quantization screen

Hypothesis: native FP8 linear computation improves prefill throughput on RTX 4080.
Change only `quantization=fp8_per_tensor` from `iteration32.json`; keep BF16
nonquantized operations, automatic (BF16) KV-cache dtype, 2,048-token budget,
two sequences, frozen 32-row order, prompts, and original merged checkpoint.
Online conversion uses per-tensor weight scales and dynamic activation scales;
verify the selected FP8 linear implementation in startup logs. No calibration
dataset or checkpoint rewrite. The fixed memory-utilization policy may allocate
more cache capacity from freed weight memory; record that consequence separately.

Use the unchanged benchmark runner, four original master/merged parity canaries,
and longest selected input before one timed pass. Retain existing canary limits
(mean score error <=0.02, max <=0.10, correlation >=0.99); stop on gate failure,
OOM, missing/nonfinite outputs, truncation, or identity drift. Do not relax gates
after observing results. Compare all paired scores/margins, threshold flips,
ranking/calibration metrics, ties, throughput, startup and telemetry against
`baseline32`, without repeating it. A >10% speedup merits further investigation,
not automatic promotion. This development slice cannot certify population quality.

Run `.venv/bin/python -m experiments.local_inference.run --config
experiments/local_inference/fp8.json` with the same offline environment. Preserve
all artifacts in `results/local_inference/fp8/`, including any failed canary.
Monitoring is active-turn only; no scheduling tool is available for later wakeups.

### FP8 result: canary rejected, no timed pass

Online per-tensor FP8 loaded and selected `CutlassFP8ScaledMMLinearKernel` on
2026-09-23. Model allocation decreased from 7.99 to 5.0 GiB; with the same 0.80
memory-utilization policy, available KV capacity was 6.31 GiB / 194,125 tokens.
This is not a claim of equivalent whole-device memory reduction: freed memory
is reused for cache. KV dtype stayed automatic/BF16; attention stayed FA2 and
gated-delta prefill stayed Triton/FLA.

The four original canaries failed the unchanged numerical gates:

| Reference | Mean absolute score error | Max error | Correlation | 0.5 flips |
| --- | ---: | ---: | ---: | ---: |
| FP32 master (eager) | 0.064793 | 0.255398 | 0.866748 | 0 |
| BF16 merged (eager) | 0.069270 | 0.255398 | 0.888389 | 0 |
| Existing BF16 vLLM canaries | 0.057170 | 0.224908 | 0.880855 | 0 |

The largest serving-to-serving change was the STRIDE positive canary: score
0.407333 -> 0.182426, with logit-margin delta -1.125. Unchanged threshold
decisions do not imply continuous-score parity. These four rows do not establish
population degradation or rule out other FP8 scaling schemes.

As predeclared, execution stopped before the longest-row canary and 32-row timed
pass; **no FP8 throughput or full-set ranking result exists**. Engine construction
took 127.68 seconds, including 42.81 seconds compilation and 58.15 seconds initial
profiling/warmup. Total failed subprocess time was 158.11 seconds, not comparable
to a successful cached-baseline total. All workers and telemetry exited.

Keep BF16 as reference. A possible next experiment is finer-grained FP8 weight
scaling or targeted layer exclusions, with a separately frozen config and the
same canary gates; neither has been run. Evidence: `fp8/serving_parity.json`,
`fp8/baseline_canary_comparison.json`, timing/telemetry artifacts under the result
root, and `logs/local/local_inference/20260923T214702Z-benchmark.log`. The campaign
status is `fp8_status.json` (failed); the child status remains at its last phase
(`loading`) because the unmodified benchmark stopped at the gate.

## Isolated BF16 MLP kernel screen

Hypothesis: another BLAS preference or Triton tile improves the two measured MLP
shapes without changing BF16 input/output or model weights. Use real merged
layer-0 weights and seeded synthetic normal activations, not an additional judge
dataset pass. Shapes are (M,N,K)=(2048,18432,2560) and (2048,2560,9216), contiguous
X[M,K] and W[N,K], computing X @ W.T without bias or activation. Baseline is
PyTorch functional.linear with default BLAS preference and existing BF16
reduced-precision-reduction allowance. Compare cuBLASLt preference, strict
cuBLAS/cuBLASLt with reduced-precision reductions disabled, and eight explicit
Triton tile configurations using FP32 accumulators. Backend preference is not
a guarantee of the selected kernel; no serving integration is authorized here.

Compile and validate all candidates first against FP32 GEMM with TF32 disabled.
Numerical screening limits: relative L2 <=0.005 and max absolute error divided
by reference RMS <=0.05; all outputs must be finite. These are synthetic kernel
screening tolerances, not judge-score acceptance criteria. Invalid candidates
are excluded; retain compilation/resource failures as negative results.
Warm GPU for three seconds, use fixed randomized candidate order, five warmup
calls, then one 256-call timed window per candidate with CUDA events and wall
timing. This averages kernel calls within one measurement, not repeated dataset
passes. Record pre/post temperatures, SM clocks, and thermal throttling. No
variance estimate or model-quality promotion is claimed. Prefer >10% speedups
for further investigation, not small single-window differences. Repeated
same-buffer cache behavior and synthetic activations limit serving relevance.

Run `.venv/bin/python -m experiments.local_inference.gemm_bench --output
results/local_inference/gemm_bench_bf16` with the baseline local environment.
Stop on OOM, nonfinite baseline, or missing measurements. Preserve prior outputs.
The existing finite default may be timed as a reference despite failing the
synthetic error guard; retain that failure explicitly, never promote it as passing.

### Result: no meaningful speedup

| Projection | Default ms | Fastest candidate ms | Apparent speedup |
| --- | ---: | ---: | ---: |
| Gate/up | 1.98465 | 1.94998 (Triton 64/128/64, 4 warps) | 1.018x |
| Down | 1.05308 | 1.02503 (Triton 128/64/64, 4 warps) | 1.027x |

Neither clears the predeclared 10% investigation threshold. GPU temperature
varied roughly 71–84 C, clocks 2520–2775 MHz, with intermittent thermal throttling.
Keep the existing serving path; no judge dataset pass or vLLM change was made.

All gate/up outputs were bit-identical to default on this input. For down,
strict PyTorch and all Triton candidates passed with relative L2 versus FP32
0.001656 and maximum error/reference RMS 0.03670. Default had 0.002334/0.05532;
cuBLASLt preference with reduced-precision reductions had 0.002025/0.05054.
Both failed only the 0.05 maximum-error guard. This synthetic guard is not a
judge-quality verdict. Strict/Triton down outputs differ from default (relative
L2 0.002625); lower error against FP32 does not establish judge-score parity.
Strict cuBLAS and cuBLASLt down times were 1.44427 and 1.31766 ms: the custom
kernel improves on those strict references, not meaningfully on production default.

Recovery disclosure: the initial runner excluded the default down timing after
its guard failure, timed ten valid alternatives, then crashed when computing a
speedup without a denominator. Their completed timing rows survive in the log.
The corrected runner measured **only the missing default down reference** in a
new process, at 80 C/2760 MHz. All regenerated validation metrics matched exactly;
no completed timing was repeated. Cross-process thermals further limit inference.
The non-strict cuBLASLt down candidate remains untimed because it failed screening.

Artifacts under `results/local_inference/`: `gemm_bench_bf16/result.json` retains
the original partial result; `gemm_bench_down_reference/` records recovery;
`gemm_bench_bf16/comparison.json` combines them with the original log at
`logs/local/local_inference/gemm_bench_bf16.log`. Reconstruct the combined record
with `.venv/bin/python -m experiments.local_inference.summarize_gemm`.

## GPU profiling diagnostic

### Shape mapping and deeper counters

Hypothesis: pairing actual `aten::mm` input shapes with CUDA kernels and deeper
scheduler/pipeline counters distinguishes expensive model projections and their
execution constraints. Capture shapes on the same eight-row, 16-step baseline
diagnostic using `--early-cupti --record-shapes`, correlating CPU operators and
GPU kernels by external ID. Map dimensions to the installed model definitions
and checkpoint weight shapes; report ambiguity rather than guessing module IDs.
Then collect six matching GEMM launches with Nsight `SchedulerStats`,
`WarpStateStats`, and `ComputeWorkloadAnalysis` alongside basic counters.
Stop on engine/capture failure, absent shape correlation, or invalid counters.
These are instrumentation-only changes: no serving optimization, throughput
comparison, repeated benchmark pass, or model-quality promotion is in scope.

Completed: the shape-aware trace captured 12,944 GPU kernels and matched all
2,448 GEMM/GEMV calls to `aten::mm` shapes through external IDs, with zero
unmatched GEMM time. Mapping uses actual checkpoint shapes plus the installed
Qwen3.5 stacked-parameter mappings (QKV/Z, gate/up, and B/A fusion). The shape
table below uses A[M,K] x B[K,N]; the prefill calls all had M=2,048 in this window.

| Projection | K | N | Share of all sampled GPU kernel time |
| --- | ---: | ---: | ---: |
| MLP fused gate/up | 2,560 | 18,432 | 30.87% |
| MLP down | 9,216 | 2,560 | 16.34% |
| Gated-delta fused QKV/Z input | 2,560 | 12,288 | 15.86% |
| Attention output (shared shape) | 4,096 | 2,560 | 6.92% |
| Full-attention QKV, including query gate | 2,560 | 10,240 | 4.32% |

The MLP pair accounts for **47.21%**. The vocabulary head uses M=1 or 2 and
contributes under 1%; it is not a major target in this workload. This is a
shape-recording trace, not a new speed measurement or proof of full-run shares.
The eight scores have mean/max absolute deltas 0.007535/0.030490 versus their
baseline32 values and zero threshold flips; selection/scheduling differs.

The deeper Nsight capture collected six launches (17 counter replay passes each).
Their kernel names, launch grids, and order match the first six matching kernels
in the shape trace: gated-delta input, attention output, MLP gate/up, MLP down,
gated-delta input, attention output. Thus the MLP rows below are explicitly
identified; these are not guessed from a CUTLASS name alone.

| Counter | MLP gate/up | MLP down |
| --- | ---: | ---: |
| Tensor-pipeline activity, % peak elapsed | 46.14% | 43.54% |
| DRAM throughput, % peak elapsed | 13.64% | 13.88% |
| Eligible warps/scheduler/active cycle | 0.103 | 0.093 |
| Scheduler issue-active cycles | 7.02% | 6.09% |
| Math-pipe-throttle cycles per issued warp instruction | 23.81 | 27.37 |

Math-pipe throttle contributes about 83% of the MLP between-instruction warp
cycles; attention-output launches show about 63%. This points to math-pipeline
pressure with limited ready-warp supply, not bulk DRAM bandwidth saturation.
Low issue rate is not itself a percentage of lost FLOPs: tensor instructions
perform substantial work and have different execution/issue rates. Similarly,
the roughly 44% SM figure is a hardware throughput aggregate, **not MFU**.
The tensor pipeline is the highest contributor in this deeper capture. Do not
interpret low occupancy or a sub-100% counter as an available proportional
speedup; stalls may reflect the selected instruction throughput and tiling.
See [NVIDIA's metric/stall definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

Recommended next intervention, not implemented: isolated BF16 GEMM comparisons
on (M,N,K)=(2048,18432,2560) and (2048,2560,9216), preserving transposed weight
layout, accumulation semantics, and numerical checks. Compare kernel algorithms
or tile shapes before integration. The installed CUDA unquantized vLLM path
dispatches to `torch.nn.functional.linear` (layers/utils.py), so do not assume a
quantized-linear backend toggle selects these BF16 kernels. Confirm any kernel
win on the frozen 32-row one-pass workload and paired scores. No kernel, weights,
precision, or serving setting was changed in this diagnostic.

Reproduce shape analysis using `python -m experiments.local_inference.shape_summary
<trace.json.gz> --output <summary.json>`. The mapping is explicitly frozen to
Gleipnir 4B and retains ambiguous attention-output dimensions as one category.
Artifacts: `results/local_inference/profile_shapes_2048/shape_summary.json` and
its raw trace; `results/local_inference/gemm_deep_2048.ncu-repz` and
`profile_deep_2048/{details.txt,counters.csv}`. Deep collection adds
`--section SchedulerStats --section WarpStateStats --section ComputeWorkloadAnalysis`
to the basic counter command below. All workers exited successfully. Replay,
uncontrolled clocks/caches, early-chunk sampling, and host-memory backup caveats
continue to apply; this is not a workload MFU measurement.

Hypothesis: a bounded GPU activity trace identifies which kernel families deserve
optimization after the prefill-budget screen showed no meaningful gain. This is
not a throughput measurement, repeat, quality promotion, or held-out evaluation.
Use unchanged 2,048-token baseline settings, warm up on the four original canaries
plus the longest 32-row input, then submit every fourth frozen row (eight rows)
and capture at most 16 engine steps after a two-step delay. Keep raw traces in
ignored results and fail if no CUDA kernel activity is present. Stop on capture
failure, OOM, invalid outputs, or engine failure; do not interpret CPU dispatch
times as GPU kernel times. Kernel duration alone does not establish compute versus
memory-bandwidth saturation; that requires additional hardware-counter evidence.

Run `.venv/bin/python -m experiments.local_inference.profile --early-cupti --output
results/local_inference/<new-profile-name>` with the same offline environment
as the baseline. No driver or system security changes are part of this diagnostic.

### Working capture and bottleneck evidence (2026-09-23)

Ordinary late PyTorch-profiler initialization repeatedly returned
`CUPTI_ERROR_UNKNOWN (999)` inside vLLM and produced CPU-only traces. A tiny
standalone PyTorch CUDA capture worked, but both delayed and immediate starts
after model warmup failed in vLLM. A diagnostic worker subclass now initializes
CUPTI with a two-kernel probe immediately after device setup, before model
loading/graph capture. The probe stops before model setup; a separate bounded
inference capture then succeeds. This is an initialization-order workaround;
the underlying CUPTI/WSL failure mechanism is not established. Production
benchmark workers and model settings were not changed.

`profile_torch_early` captured **12,944 CUDA kernels across 16 prefill steps**,
3.3339 seconds summed kernel duration in a 3.3728-second first-to-last-kernel
interval. Kernel activity occupied 98.85% of that interval. This supports GPU
kernel execution, not CPU scheduling gaps, as the main cost in the sampled
window. It does not measure full-run GPU utilization or uninstrumented latency.

| Kernel family (name-based grouping) | Share of summed GPU kernel time |
| --- | ---: |
| Matrix multiplication (GEMM/GEMV) | 75.51% |
| FlashAttention | 11.21% |
| Named gated-delta / convolution kernels | 7.29% |
| KV writes and bookkeeping | 0.12% |
| Other, including fused elementwise/normalization | 5.86% |

The single largest GEMM kernel accounts for 51.10%. KV-cache **reads** are
included in attention, not the 0.12% write/bookkeeping figure. The evidence
prioritizes dense linear-algebra execution over removing cache writes or tuning
CPU scheduling. It does not establish compute-bound versus bandwidth-bound
GEMMs, nor prove any specific kernel/precision change will improve throughput.
The capture samples early chunks of eight every-fourth rows, not every context
length in the complete workload. Scores on the eight completed diagnostic rows
had mean/max absolute differences 0.007535/0.030490 versus their baseline32
scores, with no threshold flips; scheduling differs, and this is not a parity
certification. All diagnostic workers have been stopped.

Separate tooling limits: installed Nsight Systems 2024.6.2 explicitly reports
that driver CUDA 13.4 is unsupported and its smoke report contains no GPU kernels.
Nsight Compute 2025.1.1 returns `ERR_NVGPUCTRPERM` on a one-kernel smoke test.
Hardware-counter profiling therefore needs Windows-host permission, and may
also require newer tooling. No permissions, drivers, or packages were changed.
NVIDIA documents Windows counter access under NVIDIA App > System > Advanced >
Developer > Manage GPU Performance Counters (administrator-controlled); enabling
non-admin access broadens local profiling access. See the
[official permission guidance](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters).

Artifacts: `results/local_inference/profile_torch_early/kernel_summary.json`,
`capture_validation.json`, `scores.json`, configs, frozen selection, and
`rank0.1790197209921988054.pt.trace.json.gz`. CPU-only failures are preserved in
`profile_torch_retry` and `profile_torch_immediate`; their logs remain under
`logs/local/local_inference/`. Summarize a trace with
`.venv/bin/python -m experiments.local_inference.profile_summary <trace.json.gz>
--output <summary.json>`; CPU-only traces are rejected explicitly.

### Hardware-counter permission retest

After the user restarted Windows, the native canary successfully collected
hardware counters even with the installed Nsight Compute 2025.1.1. No additional
permission or driver change was needed. `counter_after_reboot_old.log` records
the successful eight-counter-pass collection, distinct from benchmark repeats.

Next diagnostic: retain the 2,048-token baseline workload and warmup, use
`profile --kind cuda` under Nsight with `--profile-from-start off`, and collect
six launches matching the three dominant GEMM names. Hypothesis: compute versus
memory throughput counters narrow the dense-linear-algebra bottleneck. This is
kernel replay, not a throughput or quality comparison; keep clocks uncontrolled
and disable cache flushing to better retain workload cache conditions. Stop on
capture/engine failure and require a populated Nsight report before interpreting
metrics. Sampling early GEMMs is not evidence for every layer or context length.

The bounded capture completed successfully with Nsight Compute 2026.3.0:
six launches spanning the three dominant GEMM names, eight counter replay passes
per launch. DRAM throughput was **11.69–14.32%** of peak, compute (SM) throughput
**43.58–45.68%**, L2 throughput **24.22–34.04%**, and achieved occupancy
**15.96–16.71%**. All six launches had 16.67% theoretical occupancy. Kernels used
222–234 registers per thread; the 256-thread kernels were limited to one block
per SM by registers and shared memory, and the 128-thread kernel to two blocks
by registers. Nsight flags low compute/memory utilization and suggests examining
scheduler/warp-state latency. These counters argue against saturated DRAM
bandwidth in the sampled GEMMs and motivate kernel/tiling/latency-hiding analysis.
Low occupancy is a resource constraint, not by itself proof of the dominant stall
or a predicted speedup. Compute is not saturated either. No optimization was
implemented from this diagnostic.

The profile used uncontrolled clocks and caches, kernel serialization/replay,
and host-memory backup for device state. Do not use its elapsed time as a
throughput comparison or equate these metrics with the full 32-row run under
thermal throttling. Workers exited normally. Artifacts:
`results/local_inference/gemm_counters_2048.ncu-repz` and
`profile_counters_2048/{counters.csv,details.txt,scores.json,config.json}`.
The text report includes Nsight's speculative local speedup estimates; these
are not measured improvements and should not be reported as such.

The native canary is now tracked as `counter_smoke.cu` so it survives reboots.
Compile with `/usr/local/cuda-12.8/bin/nvcc -arch=sm_89 -o
/tmp/gleipnir_counter_smoke experiments/local_inference/counter_smoke.cu`.
The restored current profiler is at
`/tmp/gleipnir-ncu-restored/opt/nvidia/nsight-compute/2026.3.0/ncu`;
re-extract the cached Debian package to a whitespace-free path if needed.
For the model capture, launch `profile --kind cuda --output <new-directory>`
under that profiler with `--target-processes all --profile-from-start off
--set basic --clock-control none --cache-control none --kernel-name-base demangled
--kernel-name 'regex:.*(gemm_relu_bf16_256x128|ampere_bf16_s16816gemm_bf16_256x128|ampere_bf16_s1688gemm_bf16_128x128).*'
--launch-count 6 --export <new-report-path>`. Use the baseline offline environment.
Do not combine this CUDA/Nsight capture mode with `--early-cupti`: that would
create a competing PyTorch CUPTI subscriber. Export reports via `ncu --import
<report.ncu-repz> --page raw --csv` or `--page details`.

After the user changed Windows counter permissions, the old profiler no longer
reported the standalone `ERR_NVGPUCTRPERM` error, but failed with a driver-resource
or permission error. A current Nsight Compute 2026.3.0.13 package was downloaded
from NVIDIA's configured apt repository without installing it system-wide.
Package SHA-256:
`84feeebbe340239f3d33a9900279cacbb33a6ceb8780f65897574e0c2add9564`.
The package remains under `.cache/`; extraction was moved to
`/tmp/gleipnir-ncu-counter-tools-2026.3.0` because this profiler rejects whitespace
in its installation path, even through a symlink.

Current tooling fails with `Failed to prepare kernel for profiling` / `Unknown
error on device 0` on both a tiny PyTorch workload and a native CUDA 12.8 kernel
compiled specifically for SM89. A fresh temporary lock directory did not fix
the failure. Native CUDA execution without profiling succeeds. No competing
Nsight/DCGM/engine processes were found. Thus counter access is still unverified;
do not claim the permission change resolved it, and do not launch a full model
counter capture yet. The next proposed diagnostic is a user-controlled Windows
restart to refresh driver/WSL state, not a proven fix. A matching WSL report was
[resolved after reboot](https://forums.developer.nvidia.com/t/nsight-compute-fails-to-profile-kernels-on-wsl-windows11/287939).
No system restart, driver update, or further security change was performed.
Logs: `logs/local/local_inference/counter_smoke_2026_nospace.log`,
`counter_native_smoke.log`, and `counter_native_fresh_lock.log`.

## First optimization: 4,096-token prefill budget

Hypothesis: doubling `max_num_batched_tokens` from 2,048 to 4,096 improves
prefill efficiency by reducing chunk/scheduling overhead. Change only this
engine setting; retain the frozen 32-row order, weights, prompt, two concurrent
sequences, cache settings, canaries, and one timed pass. Compare against
`baseline32` without rerunning it. This is a development screen, not a held-out
quality claim or automatic promotion. Report paired score and margin drift,
threshold flips, metrics, and thermal conditions; stop on existing parity gates,
OOM, missing/nonfinite outputs, truncation, or artifact drift. No new numerical
equivalence tolerance or repeat-noise estimate is claimed.

Run `.venv/bin/python -m experiments.local_inference.run --config
experiments/local_inference/prefill4096.json`. Results are saved separately to
`results/local_inference/prefill4096/`, with campaign status in
`results/local_inference/prefill4096_status.json`. Existing results are protected.

Measured result (2026-09-23): **no demonstrated meaningful speedup**. One pass
scored all 338,780 tokens in 35.4894 seconds versus baseline 35.9089 seconds:
9,545.95 versus 9,434.42 prompt tokens/s, a 1.0118x throughput ratio (+1.18%).
With one pass and differing thermal histories, this small difference is
inconclusive; retain 2,048 as the reference, not an automatic promotion.

Paired score drift: mean absolute 0.005636, maximum 0.030967, p95 0.029793,
correlation 0.999574, zero threshold flips. Maximum absolute drift among the six
baseline scores in [0.4, 0.6) was 0.030967. Macro AUROC changed
0.921488 -> 0.925620; pAUROC@20 0.756198 -> 0.776860; Brier
0.097959 -> 0.099486. These small-set ranking changes are not evidence of a
better judge; numerical equivalence is not established. Unique scores were
30 -> 31. Both serving canary comparisons passed, with unchanged input/model
identities, and the longest input completed without OOM or truncation.

Whole-process time was 172.51 seconds versus 85.30 seconds. Construction took
107.48 seconds versus 24.07 seconds: this candidate needed a new compiled graph
(22.63 seconds torch.compile) and an initial profiling/warmup run (59.85 seconds),
whereas the baseline reused existing compilation caches. Do not attribute this
startup difference to steady-state scoring. Canary warmup was 6.59 seconds.
No second run was made to measure cached candidate startup.

All three scoring telemetry samples showed software thermal throttling at 84 C
and SM clocks of 2,595–2,670 MHz, compared with baseline 85–87 C. Sampled GPU
memory was 14,451 MiB (whole-device usage); vLLM reported 3.39 GiB cache capacity
or 104,261 tokens. The worker and telemetry process exited successfully.

Reproduce the paired audit with `.venv/bin/python -m
experiments.local_inference.compare_runs results/local_inference/baseline32
results/local_inference/prefill4096`. `comparison.json` retains per-row scores,
score/margin deltas, flips, per-source drift, baseline-score bins, and metrics.
Raw logprobs and launch/engine settings remain in each condition's artifacts.

## 32-row iteration workload

The user selected 32 rows for fast iteration after the initial 512-row baseline.
`iteration32.json` freezes four STRIDE negatives, six STRIDE positives, and
eleven examples of each Gloom label, selected from the existing 512 rows by the
same seeded source/label/length-bin procedure. The slice contains 338,780 prompt
tokens (2,379–27,544 per row). Preserve the 512-row artifact for occasional
broader checks. Do not interpret small-set AUROC as a precise quality estimate;
retain paired logprobs, margins, scores, threshold flips, and source coverage.

Hypothesis: the smaller workload gives a substantially faster complete iteration,
although engine startup may dominate. The intervention changes only dataset
size: retain the merged weights, serving settings, four original parity rows,
one longest-selected-row warmup, and one timed pass. Existing parity and failure
gates still apply. Measure engine initialization, canary warmup, scoring, and
whole-process wall time separately. Compilation caches are reused naturally;
this is a fresh-process measurement, not a deliberately cold-cache experiment.

Run `.venv/bin/python -m experiments.local_inference.run --iteration32`.
Inputs are `data/local_inference/iteration32.jsonl` and its manifest. Results
are `results/local_inference/baseline32/`; campaign status is
`results/local_inference/iteration32_status.json`. The ordinary invocation
retains the historical 512-row preparation workflow. To record thermals during
this run, invoke `telemetry --output results/local_inference/baseline32 --status
results/local_inference/iteration32_status.json` through the experiment module.

Measured on 2026-09-23, one pass completed with the following wall times:

| Phase | Seconds |
| --- | ---: |
| Main-function preparation/imports and artifact validation | 15.66 |
| vLLM construction (`LLM(...)`, including worker startup) | 24.07 |
| Parity and longest-input canary warmup | 4.48 |
| 32-row scoring, including rendering and output extraction | 35.91 |
| Complete benchmark subprocess, including shutdown | **85.30** |

The total includes initial module imports and shutdown outside the separately
timed main-function phases. Throughput is **9,434.42 prompt tokens/s**. Cached
compiled graphs loaded in 0.859 seconds; vLLM's narrower internal engine-init
timer was 6.98 seconds, compared with 82.48 seconds in the earlier startup.
Thus total fixed overhead is material, but cached initialization is much faster
than the earlier estimate based on the initial setup. Changing compilation
settings can produce cache misses and increase startup again.

All 32 saved identities, labels, prompt hashes, token counts, normalized scores,
and metrics were independently checked. Relative to the same rows' existing
512-row scores, mean absolute score change was 0.005573, maximum 0.030967,
correlation 0.999578, and there were zero 0.5-threshold flips. This comparison
changes scheduling/order and run conditions, not model weights. Use the fresh
32-row predictions as the iteration reference. Small-set macro pAUROC@20 was
0.756198 and AUROC 0.921488; these are not comparable with full-512 metrics as
evidence of degradation because the population differs.

Thermal samples during scoring showed 85–87 C and active software thermal
throttling; the GPU worker exited successfully. No repeat-noise estimate is
claimed. Input SHA-256:
`aadb48556b134150e43946ba39d31512498e2d61b8a88a7a24fd9617b5633f03`.
The measured totals are in `baseline32/result.json` and
`baseline32/process_timing.json`, alongside the single-pass predictions.

Hypothesis: a BF16 merge of the released Gleipnir 4B LoRA provides an efficient
fixed-weight baseline for later vLLM serving optimizations on the local RTX 4080.
The initial baseline used a frozen 512-row ID subset, retained for occasional
broader checks after selecting 32 rows for iteration. Neither full-ID
confirmation nor strict-OOD selection is in scope.

Reconstruct CoT-removed trajectories from the checksum-pinned STRIDE and Gloom
sources using the existing ID loaders and source-aware CoT remover. Preserve the
compact student prompt and non-thinking `Prediction:` boundary. Stratify by
source, hard label, and length quartile, then select by seeded ID hash. Freeze
membership before scoring. Source labels are evaluation-only; no teacher calls.
All source datasets remain ignored and must not be redistributed (the existing
source inventory records unspecified upstream licenses).

The frozen subset has 5,760,843 prompt tokens, ranging from 1,291 to 29,475
tokens per row, with SHA-256
`f5800ce52b38184bf3854fbf8e7a91257a3774c2f0a859c598c97e26f5829ebf`.
Reconstruction matches the recorded 3,012-row population's 33,750,959 prompt
tokens; its JSONL metadata schema is new, so it is not claimed to match the
historical full-file hash. The four short eager canaries have lengths
1,291, 1,354, 1,716, and 2,195; no prompt is shortened for a canary.

Download the published master adapter at a resolved immutable Hub revision and
the pinned Qwen3.5-4B base. Keep the FP32 master unchanged. Merge on CPU with
PEFT safe merging, save BF16 inference weights, and hash every artifact. A
bounded Transformers canary compares unadapted base, FP32-master adapter, and
merged weights; a matched vLLM canary then checks serving agreement. These
eager canary includes one short actual subset row per source/label pair; vLLM
also runs the longest subset input as a separate memory/kernel canary. Require
nonzero adapter effect, mean absolute score error <= 0.02, maximum error <= 0.10,
and correlation >= 0.99 when scores have nonzero variance. Report all flips;
these are baseline-construction tolerances, not future optimization tolerances.
Stop before full-subset evaluation if a gate fails.

Use one persistent vLLM engine, one constrained decision token, explicitly
requested 0/1 logprobs, and one warmed timed pass over the fixed input order.
Disable prefix caching initially to avoid replay-dependent timings. Record raw
decision logprobs, margins, scores, per-source/macro ranking and calibration,
threshold diagnostics, ties, initialization time, and prompt tokens/s.
Stop on missing or nonfinite scores, OOM, truncation, artifact/hash drift, or a
failed parity gate. Do not change data or quality criteria in response to scores.
Future optimizations will be compared against this merged baseline on these
same rows. Per the user's update during the first pass, use one measurement per
condition and target meaningful speedups; repeat noise is not estimated. Small
speed differences should not be interpreted as improvements. Keep paired score
drift diagnostics even with single-pass timing.

The workstation has a 16 GB RTX 4080 and 47 GiB RAM. No Slurm or cloud capacity
is involved. No in-chat scheduling tool is available in this session; monitoring
uses active-turn startup checks and continued active-turn checks during scoring,
and cannot promise agent follow-ups after the turn ends.

Startup exposed two launcher/build issues: explicit Python invocation did not
put `.venv/bin/ninja` on PATH, and FlashInfer's sampling Ninja file does not
escape the space in this checkout's `AI Control` directory. The runner now
adds its executable directory to PATH. The frozen environment sets
`VLLM_USE_FLASHINFER_SAMPLER=0`; this workload uses greedy one-token decisions,
and vLLM's native sampling path avoids that optional JIT dependency. FlashAttention
2 and Triton/FLA GDN remain the automatic model-kernel choices on SM89. Both
failed startup attempts and logs are retained separately.

Provenance: reuse `prepare_qwen_reasoning_id_benchmark` source loaders,
`id_cot_only_evaluation.prepare.strip_cot`, the shared student prompt, and shared
binary metric functions. This is a separate local campaign; historical configs
and result contracts are unchanged.

## Run

```bash
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/python -m experiments.local_inference.fetch
.venv/bin/python -m experiments.local_inference.run
```

The runner prepares the subset, merges the release, checks the eager reference,
and starts the vLLM baseline sequentially. Each phase has a separate log under
`logs/local/local_inference/`; status and results are in
`results/local_inference/`. Failed attempts must be retained before rerunning
the baseline; completed timed passes are not silently overwritten or resumed.

For local temperature/clock diagnostics, run
`.venv/bin/python -m experiments.local_inference.telemetry` in a second process
after `baseline/status.json` exists. It samples every ten seconds and exits when
the campaign completes or fails. This is instrument logging, not scheduled agent
monitoring. The first baseline showed software thermal throttling at 87 C;
compare future configurations under comparable cooling and record telemetry.
Sampling began partway through its first pass, so those samples are not a full
startup or first-pass thermal trace.

## Baseline result (2026-09-23)

One complete pass processed all 512 rows and 5,760,843 prompt tokens in
**631.65 seconds (10m32s)**: **9,120.28 prompt tokens/s**, about 0.81 rows/s.
This is wall time around `LLM.generate`, including its input rendering and
output extraction; it excludes initialization and canary warmup. Prefix caching
was disabled. The run used two concurrent sequences and a 2,048-token prefill
budget. No repeated-pass noise estimate was made.

| Source | Rows | pAUROC@20 | AUROC | Brier |
| --- | ---: | ---: | ---: | ---: |
| STRIDE | 161 | 0.967930 | 0.993197 | 0.038498 |
| Gloom-Exfiltration | 351 | 0.752760 | 0.923961 | 0.121559 |
| Source macro | 512 | 0.860345 | 0.958579 | 0.080028 |

The 512 predictions contain 397 distinct scores (115 extra tied rows). These
are subset results, not a measured improvement over the historical full-ID
metrics. All IDs, sources, labels, prompt hashes, token counts, decision-logprob
normalizations, throughput, and aggregate metrics were independently rechecked
from the saved first-pass artifacts.

The four-row eager merge gate had mean absolute score error 0.004477 and maximum
0.017909 against the original FP32 master. vLLM versus master had mean error
0.008683, maximum 0.030490, and correlation 0.999590; neither canary comparison
flipped a decision at 0.5. The longest 29,475-token subset prompt also passed.
This is bounded canary evidence, not a full-subset master-versus-merge comparison.

The GPU was thermally limited: all 47 recorded first-pass telemetry samples
reported software thermal throttling, with peak temperature 89 C and SM clocks
2,520–2,655 MHz. Hardware thermal slowdown was not observed in these samples.
Use comparable cooling/thermal conditions for future timing comparisons.

The initial runner had three passes configured; the user reduced the scope
during pass one. A task-specific watcher stopped the worker immediately after
`repeat_0.json` was saved. Only one completed pass is retained. The original
launch config and interruption log are preserved; `summarize.py` audits and
finalizes this specific saved-pass recovery. The interruption therefore does
not represent an inference failure. The checked-in config now requests one
pass directly, and single-pass repeat stability is recorded as null.

Artifacts: `results/local_inference/baseline/result.json`,
`predictions_0.json`, `repeat_0.json`, `serving_parity.json`,
`longest_canary.json`, `engine_config.json`, and `gpu_telemetry.jsonl`.
The merged model is `results/local_inference/merged_bf16/`; the original master
remains in `results/local_inference/adapter/`. Published adapter revision:
`411bdb7cf28bf153d03820f242c7f0668762fa6d`.
