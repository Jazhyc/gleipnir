# Direct FROST bindings in FP4 LoRA training

2026-10-07. Direct positional dispatch preserves native and full-model numerics,
but gives no measurable end-to-end training speedup on the frozen 320-row cohort.
Pooled controls average 3.39498 s/update; direct bindings take 3.39585 s/update
(0.026% slower). Keep the path opt-in and the selected training baseline unchanged.

## Intervention and numerical evidence

`gleipnir.kernels.fp4.frost_bindings.training_frost_bindings` bypasses NVIDIA's
per-call tensor/name/UID resolver for the selected six-operand, zero-workspace
FP4 row-descaling plans. It preserves the original packing, allocation, views,
BF16 rounding, lowered executor guards and registered autograd. It caches no
activation or adapter values. Original mode restores the unwrapped control;
context exit restores both the class method and all plan executors.

`native03` uses one compiled Transformers/PEFT Qwen3.5-4B MLP with seeded nonzero
rank-128/alpha-256 FP32 LoRA masters. Inputs are BF16. The native selected recipe
uses hardware packing and fused descale for frozen forward and input gradients.
All three token counts (193/4096/16384) produce bitwise-equal outputs and all
seven gradients: input plus six adapter parameters. Outputs/gradients are finite,
output storage is independent, changed-input and changed-adapter graph replay
is exact and actually changes outputs, and a second CUDA stream agrees exactly.
All four forward/dgrad K/N geometries execute through direct dispatch. The
context closes in original mode, with 348 direct calls across four plans.
This establishes binding correctness against the selected quantized MLP;
it does not turn historical failed BF16/FP4 recipe comparisons into passes.

## Warm complete-MLP timings

Use six warmups per mode, then ten alternating synchronized forward/backward
samples per mode. Include allocation, activation/gradient packing, adapter
contractions and all seven gradients. Automatic Inductor CUDA graphs are off
in both legs so the host binding path executes; explicit graph replay is a
separate numerical gate. This is synthetic operator timing, without GDN/FA4,
packing/batching, clipping, optimizer, scheduler or model-quality evaluation.

| Tokens | Original median, ms | Direct median, ms | Median time change | All-sample mean change | Faster pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 193 | 1.42025 | 1.40223 | -1.27% | -1.93% | 9/10 |
| 4096 | 1.48845 | 1.42627 | -4.18% | -21.77% | 10/10 |
| 16384 | 3.63240 | 3.60852 | -0.66% | -0.38% | 9/10 |

The first 4096-token control sample is 5.09956 ms; remaining control samples
are 1.43302--1.52052 ms. Preserve it and the direct leg's 1.60563-ms first sample.
Its cause is not isolated, so the 21.77% mean reduction is not evidence of a
repeatable gain of that size. Median paired reductions are 2.34/4.09/0.68%.
These operator timings do not establish a whole-training gain or attribute the
full-model bottleneck. The matched optimizer-update screen below measures the
complete pipeline and does not meet the frozen >=2% promotion threshold.

## Full optimizer-update screen

`results/b200_mlp_gemm/frostresident03/` holds the completed screen on the same
EU-RO-1 B200. Reuse the established 320 rows, explicit FP32 initialization,
LR 5e-5, logical batch 32, 147 physical partitions and selected FP4-MLP/BF16-
FlashQLA/FA4 recipe. One resident worker runs two controls, direct bindings and
a restored control. Reset masters, optimizer/scheduler, RNG and data order
between twenty-update trials; time all ten updates 11--20 per trial. Include
forward/backward, finite-gradient checks, clipping and optimizer/scheduler work;
exclude loading, validation, priming and adapter export. No profiler is active.

| Trial | Warm mean, s/update | Actual tokens/s |
| --- | ---: | ---: |
| Original control | 3.38778 | 38,796 |
| Reset repeat | 3.39819 | 38,677 |
| Direct bindings | 3.39585 | 38,704 |
| Restored control | 3.39896 | 38,669 |
| Pooled controls | 3.39498 | 38,714 |

Each timed leg processes 1,314,331 tokens. Keep every sample; the candidate sits
inside the range of the three control means. Its 0.877 ms/update difference is
0.026% slower, not a useful speedup. All timed updates have zero new native
plans, Triton specializations, Dynamo graphs and Inductor cache misses.

The first-logical-batch gate produces bitwise-equal loss (0.6376177072525024)
and every adapter gradient, with finite/missing-gradient checks, unchanged
masters and identical partitions. All four twenty-update loss histories and
final FP32 master digests also agree exactly. Direct dispatch executes 889
host calls in the targeted gate and 18,669 in training across all four K/N
geometries; this is an exercised intervention, not a no-op hidden by graph
replay. The full screen passes numerical/workload gates but fails the speed
selection criterion. No precision, training default or quality claim changes.

Initial EU setup lacks the frozen cohort/initialization and training-only
FlashQLA/convolution dependencies. The user explicitly authorizes transferring
these artifacts to this pod. Retain the interrupted workspace-copy installer
and both rejected startups: `frostresident01` imports Quack 0.5.0 from the base
serving runtime, incompatible with selected CUTLASS 4.8; restore the frozen
Quack 0.6.5 overlay. `frostresident02` rejects the active runtime-M kernel
source generation against historical hashes in the legacy profile helper.
Use the standard recipe's existing checksum-bound runtime-shape equivalence
proofs, checking both historical and active fingerprints; preserve the
historical strict numerical failures. `frostresident03` then completes.

Untimed priming takes 1,491.964 s. New variants take tens to hundreds of seconds;
reused batches take about 2--5 s. Stage FlashQLA on local disk with a persistent
47.3 MB dependency archive, and reuse the byte-matched staged FA4 overlay during
priming while retaining the workspace original. Do not attribute a speed ratio
to this storage change: different warmup batches encounter different cache keys.
The worker environment already requests 16 compiler/build workers. Actual pod
limits are 27.2 CPU equivalents and approximately 301 GiB RAM; RAM is not the
limiter. These settings do not turn every demand-driven backend JIT call into
parallel compilation. Future setup overlaps the independent FLA, convolution
and FlashQLA jobs and launches against the verified local runtime; timed model
trials remain sequential. Two concurrency/failure tests and 29 resident/runtime
reuse checks pass; 36 source-generation/reference checks and Ruff pass.

All 240 full-screen artifacts, source snapshots, failures and logs verify
against collected checksums. Closure retains worker PID 22966 idle with the
model and shared caches on pod `qobmmj1weyevg1`; inspect `worker.json` before
reuse. vLLM remains stopped. Significant inherited warnings include experimental
BLAS preference, FA4 AuxData adaptation, absent excluded vision modules in the
text-only model, and unauthenticated HF metadata reads. Stack inspection fails
because the container blocks debugger attachment; it does not alter the run.

## Native-screen runtime, failed attempts and collection

User-authorized vLLM retirement verifies API 20025 / engine 20048 identities
and stops both processes before the probe. Pod `qobmmj1weyevg1` remains running
in EU-RO-1. The physical GPU is B200
`GPU-a2b24934-cf31-dd9e-91b5-397e49513463`, driver 580.178.04. Native runtime is
Torch 2.11.0+cu130, CUDA 13.0; the retained isolated Triton/CUTLASS/cuDNN overlays
and persistent compiler caches are reused. All five selected native source
fingerprints remain valid.

`native01` stops on the pre-refactor import layout before numerical checks.
The user then explicitly requests replacing the GPU source. Synchronize and
verify the current 176 source files; archive/remove 139 obsolete Python files,
without replacing datasets, weights, dependency environments or compiler caches.
The sync tool's unanchored `data/` exclusion had omitted `src/gleipnir/data`;
anchor data/results/log exclusions to the repository root. A real local-rsync
regression test verifies code inclusion and artifact/secret exclusions.

`native02` stops during CUDA backward capture. Raw output tensors retained
AccumulateGrad nodes created on the default stream, causing a gradient-stream
mismatch and `cudaErrorStreamCaptureImplicit`. Release those autograd graph
references before alternate-stream execution; warm/capture on an explicit
stream and retain detached captured outputs. `native03` passes all unchanged
numerical/replay gates with no stream-mismatch warning. Preserve both failures.
The inherited experimental `preferred_blas_library` warning remains visible.

Artifacts are under `results/b200_frost_training/`, including all three probes,
launch/retirement receipts, source synchronization manifest and obsolete-source
archive, raw timing samples and closure. All 52 remote artifact checksums verify
locally. Derived `native03/timing_summary.json` and CPU check logs are retained
separately. Forty-seven focused binding/transport tests and Ruff pass. The earlier native-screen closure
has no GPU processes and 0 MiB in use; vLLM remains stopped, the B200 is retained,
and no training recipe or quality selection is promoted.
