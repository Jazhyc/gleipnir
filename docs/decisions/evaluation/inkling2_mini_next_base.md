# Inkling 2 Mini as the next large-monitor base

Status: current.

User selection, 2026-11-23: `thinkingmachines/Inkling-2-Mini` (revision
`7a67ae23…`) is the base candidate to succeed Qwen3.8-27B, the base of
Gleipnir 27B. No other candidate is under consideration: Flint was ruled out
on 2026-11-20 ([experiment](../../../experiments/evaluation/flint_ood/README.md)).

**Evidence** ([experiment](../../../experiments/evaluation/inkling_ood/README.md)), all on
the unadapted checkpoint, which ships only in NVFP4/MXFP8:
- **Quality.** Strict OOD Mean-OOD pAUROC@20 0.8842 at the direct-logit
  boundary (effort none), against 0.8901 for unadapted Qwen3.8-27B; ahead on
  AgentDojo and IaC, behind on SHADE-Arena and in calibration.
- **Speed.** On the 96 agent trajectories, 4.3 requests/s one at a time and
  5.6 at 8 concurrent, against 4.1 and 4.3 for the optimized Gleipnir 27B,
  with no Inkling-specific optimization. The kernel profile puts about half
  of GPU time in memory-bound glue kernels, so fusion is the likely source of
  further speed.

**Open before it can replace the 27B.** None of these is settled:
- vLLM support exists only in nightlies (`0.34.0rc1.dev1201+gbfe3d7e87`
  used), and batched scores depend on batch composition (up to 5 nats), so
  scoring ran one sequence at a time.
- The model always prefers to open a thinking block at this boundary; the
  student would be trained at a boundary it does not use unprompted.
- ID was not scored, no adapter has been trained, and training support for
  its looped architecture and low-precision weights is unverified.
