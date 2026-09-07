# Two-GPU mixed-corpus MIL throughput

Date: 2026-09-07. Systems screen completed; full scientific experiment launched.

## Finding

On the two reserved H100 SXM5 80GB GPUs, eager two-process DDP with FLA and
causal-conv1d outperformed the compiled single-GPU MIL recipe on a matched
256-parent mixture. Both arms used eight optimizer updates, the same seed and
global batch32, NF4 double quantization/BF16 decoder compute, FP32 LoRA adapters
and selected classification projection, and non-reentrant all-layer checkpointing.

| Configuration | Mean seconds/update, steps 3–8 | Eight-step training runtime |
| --- | ---: | ---: |
| One GPU, compiled linear shells, accumulation32 | 16.5840 | 142.5314 s |
| Two GPUs, eager shells, accumulation16/rank | 10.4542 | 86.3027 s |

The steady comparison is **1.5863x throughput / 36.9622% less wall time**.
This is a comparison of usable configurations, not a pure DDP-only ablation.
It does not imply lower GPU-hours: two GPUs consume about 1.26x as many GPU-hours
at this measured speedup. The eight-step screen is a short systems estimate,
not a guarantee for every trajectory mix or cold start.

The sample contains 30 monitoring and 226 deception parents, deterministically
interleaved by identity hash. A separate 32-parent longest-context preflight
passed. Both distributed endpoints had exactly matching trainable parameters
across ranks. Final screen mean losses were .599509 (single) and .600943 (DDP);
these are training diagnostics, not held-out model-quality results.

## Correctness and startup findings

- MIL's decoder-direct projection must execute inside DDP.forward so the
  reducer sees the backward graph. Preserve the PEFT class and adapter format.
- Accelerate's outer autocast must not silently move the selected FP32 head
  projection into BF16. The dispatch disables outer autocast; the decoder owns
  its explicit BF16 scope.
- Two-process Gloo tests compare gradients and actual Trainer/Accelerate
  updates with an unsharded reference, including partial accumulation windows.
  A deliberately divergent replica is rejected. Sixty focused tests passed.
- Compiled DDP failed the existing same-weight canary before training on both
  ranks: eager `[7.65625, 9.0625]`, compiled `[7.8125, 9.1875]`. Rank-local caches
  did not resolve it. Tolerances were not widened; the promoted DDP recipe is
  explicitly eager. Its same-weight check is eager repeatability only.
- Newly empty rank-local Triton caches caused substantial autotuning throughout
  the first eight updates. Read-only worker stack samples caught FLA l2norm
  autotuning and fused gated-norm kernel compilation. Seed compatible existing
  cache entries without overwriting rank-local entries, then isolate writes.
- Membership manifests intentionally preserve source order. An interleaved
  systems screen must materialize that order, not just reorder membership keys.
  Attempts 1/2 are not used for throughput conclusions; attempt3 was cold and
  preceded the FP32 dispatch correction. Attempt4 is the accepted comparison.

## Full-run decision and audit

Launch a fresh three-epoch seed0 run at LR2e-5 on the exact 1,738 monitoring
parents plus all 13,149 deception examples. Monitoring retains `.25 * MIL_BCE`
plus Kimi final-boundary BCE; deception receives only the final-boundary loss.
Auxiliary loss is summed over eligible parents and divided by all parents, so
changing batch composition does not change a monitoring parent's weight.

DDP keeps global batch32 and 1,398 optimizer updates. Standard even-batch
sharding repeats one parent per epoch to make 14,888 distributed exposures;
the final update contains eight rather than seven parents. This small padding
difference is recorded, not hidden. The full run's initial training-only estimate
is 4.06 hours plus startup/cold shapes, versus 6.44 hours from the single-GPU
screen. Serving parity and the unchanged 3,012-row ID evaluation follow training.
No OOD consultation or new scientific arms were added.

Artifacts: `results/monitoring_mil_mixture/ddp_screen_attempt4/` contains the
accepted jobs, materialized selection, metadata and status. The frozen full-run
contract is `results/monitoring_mil_mixture_ddp/manifest.json`; original stopped
single-GPU artifacts and all failed screens remain preserved. Implementation
commit: `ab69285`. No agent scheduling tool is available in this session; the
remote pipeline and evaluation watchdog are not agent monitoring heartbeats.
