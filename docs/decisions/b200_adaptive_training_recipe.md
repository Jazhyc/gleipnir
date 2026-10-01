# Adaptive batching for future B200 training

Status: selected by explicit user instruction, 2026-10-01.

## Decision

Use selectively compiled adaptive physical batching for future single-B200
Qwen3.5-4B training with the tested per-example monitoring loss. The user selected
this recipe after reviewing the [matched execution audit](../findings/b200_execution_audit.md)
and its approximately 45% lower compiled training-step time. This is an explicit
systems recipe choice based on bounded evidence, not a claim that the original
gradient-parity screen passed or that held-out quality equivalence is established.

The reusable Hydra profile is
`systems_screen@_global_: qwen35_4b_b200_adaptive`. It inherits the tested B200
backbone, quantization, SDPA, FLA, selective compilation and twelve-layer
checkpoint settings from `qwen35_4b_b200_fast`, changing physical batching to:

- logical batch 32 with Trainer gradient accumulation 1;
- 16,384 padded tokens and physical maximum 8, with observed sizes 1, 2, 4 and 8;
- oversized traces retained as singletons at the 29,696-token context cap;
- each physical loss weighted by its example count over the logical batch;
- one clip, AdamW step and scheduler step after the whole logical batch;
- synchronized per-microbatch profiling disabled for ordinary training.

Retain rank-128/alpha-256 FP32 master adapters, NF4 double quantization/BF16
compute, AdamW at 5e-5, norm-1 clipping, linear scheduling and 3% warmup unless
a future experiment explicitly changes them. "SGD robustness" here is a
hypothesis about stochastic minibatch optimization; the optimizer remains AdamW.
The small observed loss gaps are consistent with practical tolerance of these
numerical differences, but stochastic sampling does not establish that a
persistent numerical bias will disappear. No robustness result was measured.

## Evidence and limits

In the shared warm-process ten-step diagnostic, compiled singleton averaged
18.78 seconds per step and compiled adaptive 10.30 seconds (45.2% less time,
1.82 times the throughput for training-step work). Their final common training
probe losses were 0.93897 and 0.94005, a 0.12% difference. Across all four
eager/compiled and singleton/adaptive cases the final mean-loss spread was 1.35%.
These timings exclude setup, evaluation, resets and exports; no new epoch ETA
is asserted. Larger per-example differences and distinct update vectors remain.

Preserve historical profiles, frozen contracts, failed diagnostics and the old
0.05 gradient threshold. The old systems-screen runner continues to enforce
that threshold; this decision does not silently change its preflight behavior.
Before launching a future campaign, freeze its own validation/stop conditions,
check memory and kernels for its actual data/rank/context, and use the agreed
CoT-removed ID validation for checkpoint selection. Recipe selection here does
not launch a run or select a model checkpoint. Retain the fixed-batch profile
for reproducing the earlier control.

Validation: Hydra composition confirms the selected logical/physical batch
settings, disabled profiling and AdamW, with all inherited non-batching recipe
fields and the historical screen preflight unchanged. No GPU run was started
for this configuration/documentation decision.
