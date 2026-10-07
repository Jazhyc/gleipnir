# Direct FROST bindings in FP4 LoRA training

2026-10-07. Direct positional dispatch passes the native complete-MLP gates
without changing outputs or gradients. The measured gain is modest and depends
on physical token count. Keep this as an opt-in path: full optimizer-update
throughput has not been measured and the training default is unchanged.

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
No whole-training gain is established. The largest-token result suggests
limited benefit when GPU work dominates, but these timings alone do not
attribute the bottleneck or predict a full-model speedup. Reuse native evidence
for compatible follow-ups; a full-model promotion still requires its matched
warmed-update and exact loss/gradient/update gates.

## Runtime, failed attempts and collection

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
separately. Forty-seven focused binding/transport tests and Ruff pass. Closure
has no GPU processes and 0 MiB in use; vLLM remains stopped, the B200 is retained,
and no training recipe or quality selection is promoted.
