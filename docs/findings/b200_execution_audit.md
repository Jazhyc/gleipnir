# B200 gradients, actual updates and short learning curves

Date: 2026-10-01. Status: all three user-authorized checks completed on the
existing B200. Close short-run losses coexist with different gradients and
parameter updates. The original gradient gate and selected recipe are unchanged.

## Contract and provenance

The user explicitly authorized fixed-physical-batch eager/compiled gradient
comparisons, actual clipped AdamW updates, and ten matched optimizer steps.
The [predeclared contract](../../experiments/b200_adaptive_microbatching/README.md#authorized-gradient-actual-update-and-ten-step-execution-audit)
allows this bounded diagnostic to proceed despite the failed earlier parity
screen. It does not authorize promotion, full training, or held-out selection.

Source `5767c3b56eb57519ef6ecc4de2beaf5a1ee28f30` passed **49 focused CPU tests**
covering tensor comparisons, restoration, real Trainer AdamW/scheduler behavior,
microbatch weighting and logical update boundaries. The four conditions cross
eager/selectively compiled execution with physical singleton/adaptive batching.
All use Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
rank-128 FP32 adapters, NF4 double quantization/BF16 compute, SDPA, pinned FLA
0.5.2, and the selected twelve-layer checkpoint policy. Compiled execution
retains the existing `same_as_forward` backward-autocast assumption; this audit
does not resolve the separate compiler-autocast question.

The runner verified these SHA-256 identities before launch:

| Input | SHA-256 |
| --- | --- |
| Student rows | `f9b0a322f277991edae2510b658a2298c0c585d402b39f36d6ac86352bf1a76b` |
| Soft targets | `50f163f88b42ff8b6fc04d5ff1334766d6326895ad89c87501a93eeaeae137ee` |
| Matched 320-row selection | `a27b8b3d702436839869fda2aa612ad61a4c3ef8aa9f56b4c57a4914b0a50e06` |
| Initial FP32 masters | `e2944ee2eb7be34c86ccc0e82e75ac282158cabc101258c2db15ea5eed432a49` |

Every probe and trajectory starts with the same verified master hash and empty
optimizer state. All four trajectories use the same seed-0 permutation,
32-example membership per logical update, learning rates and actual tokens.
There were **44 optimizer step calls**: four nonzero-LR probe updates and forty
trajectory steps. The first step of each trajectory has LR zero under the
unchanged 3% warmup/linear schedule, leaving 40 nonzero-LR calls overall.

The eight probes are the longest rows **within this matched 320-row cohort**,
tail-truncated to `[2048,2048,2048,2048,1024,512,256,128]`. They differ from the
historical global-longest-eight diagnostic inputs. This run does not erase or
explain the earlier 0.52–0.75 compiled batching discrepancies. Common losses
below always use one eager singleton evaluation path on these training probes,
with the model in evaluation mode. They are not held-out quality measurements.

## Same-weight gradients and actual AdamW updates

All comparisons cover all **169,869,312 trainable elements**. Relative L2 uses
the first condition as reference. The actual probe update uses fresh Trainer
AdamW state, clipping at norm 1, LR 5e-5, and no scheduler so it is nonzero.

| Comparison | Gradient relative L2 | Gradient cosine | Update relative L2 | Update cosine |
| --- | ---: | ---: | ---: | ---: |
| Eager → compiled, singleton | 0.110646 | 0.993861 | 0.337231 | 0.943138 |
| Eager → compiled, batch 8 | 0.102047 | 0.994780 | 0.324336 | 0.947403 |
| Singleton → batch 8, eager | 0.100192 | 0.994968 | 0.314035 | 0.950691 |
| Singleton → batch 8, compiled | 0.095520 | 0.995432 | 0.320874 | 0.948520 |

Identical-shape singleton repeats have relative gradient L2 0.004898 eager and
0.004535 compiled, with cosine above 0.999988. Both batch-8 repeats produce
exactly identical gradient vectors. Large repeat-to-repeat variability was not
reproduced on these inputs. Cross-backend gradient sign disagreement is 2.94%
for singleton and 2.72% for batch 8, counting coordinates with nonzero values.
Every cross-backend/partition gradient comparison still misses the original
0.05 gate.

The update norms are nearly identical (approximately 0.49650), but directions
differ. AdamW's coordinate normalization makes small-gradient sign differences
matter; gradient-relative error need not equal update-relative error. Neither
percentage is a percentage loss or quality degradation.

| Condition | Execution-path loss before update | Common eager loss after update |
| --- | ---: | ---: |
| Eager singleton | 1.246726 | 0.635620 |
| Eager batch 8 | 1.247715 | 0.646652 |
| Compiled singleton | 1.236747 | 0.638838 |
| Compiled batch 8 | 1.239383 | 0.643279 |

All common pre-update losses equal 1.246726. At fixed physical batching,
post-update eager/compiled mean losses differ by about 0.5%, despite 32–34%
relative update L2 differences. Individual losses are recorded separately;
the mean improvement does not imply every probe improves.

## Ten matched logical updates

Each trajectory consumes all 320 selected training rows, **1,314,331 actual
tokens**, untruncated at the existing 29,696-token cap. Adaptive batching uses
16,384 padded tokens/max 8, realizes physical sizes **1, 2, 4 and 8**, and retains
oversized traces as singletons. It materializes 1,411,298 padded tokens, 7.38%
above the actual-token count. Normalization remains an equally weighted mean
over each logical batch of 32, with one clip/AdamW/scheduler boundary per step.

| Condition | Final common probe loss | Mean of ten pre-update training batch losses | Sum of training-step seconds |
| --- | ---: | ---: | ---: |
| Eager singleton | 0.927562 | 0.688027 | 216.61 |
| Eager adaptive | 0.930062 | 0.688972 | 134.29 |
| Compiled singleton | 0.938970 | 0.692253 | 187.82 |
| Compiled adaptive | 0.940055 | 0.687012 | 102.99 |

The final mean probe-loss spread is **1.35%** across all four conditions.
Compiled loss is 1.23% above eager for singletons and 1.07% above eager for
adaptive batching. Batching changes final loss by 0.27% eager and 0.12% compiled.
The maximum fixed-partition eager/compiled mean-loss gap during the ten steps is
2.09%. Final per-example gaps are larger: up to 6.91% (absolute loss 0.06404)
for eager singleton versus adaptive. Close means alone are insufficient.

Cumulative ten-step update vectors remain different: relative L2 is 0.27454
for eager/compiled singleton and 0.27682 for eager/compiled adaptive; within
each backend singleton/adaptive error is 0.32821 eager and 0.22619 compiled.
Thus similar losses do not mean identical learning trajectories or adapters.

The timing column measures synchronized training-step work only. Common-probe
evaluation, state resets, setup and master export are excluded, and conditions
share one process with previously warmed compilation caches. Adaptive compute
time is promising, but these measurements are **not a validated complete-loop
throughput gain or revised epoch ETA**. The earlier provisional 3.52-hour epoch
compute estimate remains unchanged.

## Interpretation, artifacts and decision

All runs completed without OOM or nonfinite/runtime failure. Logs contain no
recompile-limit or unexpected fallback warnings. Compiler counters are cumulative
across the process: eager trajectories inherit the earlier compiled-probe
counters and must not be interpreted as compiled. The final count is eleven
unique graphs; graph breaks at deliberately disabled native linear kernels and
Accelerate hooks are expected boundaries of this selective recipe.

On this bounded workload, gradient/update L2 alone substantially overstates the
observed difference in mean losses. The results support the user's suggestion
to examine actual optimization rather than infer broken kernels from gradient
L2 alone. They do not prove long-run or held-out quality parity, identify the
operator causing the numerical differences, or resolve the earlier stress test.
Keep the selected fixed-batch recipe and the failed campaign's original gate.
Adaptive batching remains an opt-in candidate for a separately matched
throughput and quality experiment.

Local collected artifacts are `results/b200_execution_audit/contract.json`,
`execution_audit.json`, derived `summary.json`, `learning_curves.png` and
`learning_curves.svg`, plus `logs/runpod/b200_execution_audit/launcher.log`.
The complete metrics include per-step/per-example losses, membership,
partitions, gradients, updates, learning rates and compiler counters. Four final
diagnostic FP32 master checkpoints remain on the Pod's persistent volume under
`/workspace/gleipnir/results/b200_execution_audit/`; no serving export was made.
The audit restores the original masters and model methods on exit, with scoped
restoration covered by CPU tests. No held-out evaluation or recipe selection ran.

The GPU is idle after completion. Pod `alzfug70g5237b` remains running as requested
at the live-verified **$6.79/hour**; no new capacity was provisioned. Active-turn
monitoring covered completion; no after-turn follow-up is claimed.
