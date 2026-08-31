# MATS project log

This log records personal project time for the 20-hour MATS application work
on tool-trajectory monitoring. Session durations are self-reported; commit
timestamps are supporting evidence for what happened, not the source of the
hour count. Sparse autonomous-agent commits after a session may be recorded as
follow-up outcomes but do not add personal time.

## Time summary

| Date | Session | Personal time | Cumulative time | Remaining |
| --- | --- | ---: | ---: | ---: |
| 2026-08-30 | 1 | 6 hours | 6 hours | 14 hours |
| 2026-08-30 | 2 | 5 hours | 11 hours | 9 hours |
| 2026-08-31 | 3 | 2 hours | 13 hours | 7 hours |

## 2026-08-30 — Session 1

**Time:** approximately 00:00–06:00, 6 hours

**Commit evidence:** `11dabbf` through `946eee9`, committed 00:28–05:19

Established the action-only tool-trajectory monitoring workstream based on
Sinha et al.'s deliberative-monitoring paper:

- Defined the research question, dataset and split contract, evaluation
  protocol, held-out selection rule, stop conditions, and versioned binary
  teacher/student prompts.
- Built resumable, prompt-aware OpenRouter teacher-logit caching with provider
  pinning, price ceilings, batched checkpoints, provider-separated shards, raw
  decision-token logprobs, token usage, and provenance.
- Probed and documented the reserved Lambda H100 environment and pinned the
  Qwen3.5, vLLM, and FLA execution contract needed for later training.
- Inspected and materialized the paper data: an 8,688-row balanced training
  mixture, 3,012 held-out in-distribution examples, and a separately protected
  six-source OOD suite.
- Ran Kimi K3 output-interface, provider-cache, and structured-decoding
  canaries. The initial balanced canary reached `0.9746` pooled AUROC. The
  probes showed that cross-row prefix caching was ineffective and that strict
  structured decoding changed the teacher's soft distribution.
- Compared immediate Qwen decision logits with visible and native reasoning.
  Immediate logits were the stronger and better-calibrated baseline, including
  on the held-out in-distribution data.
- Completed the 8,688-row Kimi teacher cache, implemented the paper's raw
  normalized pAUROC@20 metric, documented annotation and inference economics,
  and published a sanitized logit-only export. On the training mixture, Kimi
  reached `0.9914` mean-source AUROC and `0.9636` mean-source pAUROC@20.
- Initiated the Qwen3.5-9B Kimi-soft data-scaling campaign.

### Uncounted follow-up

Autonomous follow-up commits `4ca94fd` through `f408b2a`, made from 14:07 to
14:52, implemented and repaired the scaling launcher and finalized a one-epoch
recipe. The campaign contains six nested tool-trajectory sizes from 204 through
8,688 examples and one combined tool-trajectory/Aletheia condition. These
commits are an outcome of the session but add no personal time to the log.

## 2026-08-30 — Session 2

**Time:** approximately 20:00–00:00, plus 1 hour of later result review;
5 hours total

**Commit evidence:** initial work in `c6962e1` through `4958ed8`, committed
20:23–23:55. Autonomous result records reviewed during the extension:
`9a6d83b`, `dfb31ae`, and `a01ed7a`.

Established frozen baselines and ablations while the main scaling campaign was
running:

- Materialized and audited the full 6,395-example, six-source strict-OOD suite.
- Completed the Kimi K3 binary-logit OOD baseline: `0.9084` Mean-OOD
  pAUROC@20 at a projected uncached cost of `$25.1177` per 1,000 evaluations.
- Established unadapted full-prompt baselines using persistent vLLM engines and
  constrained decision-token logprobs: Qwen3.5-9B reached `0.6976`,
  Qwen3.5-4B reached `0.6175`, and GPT-OSS-120B reached `0.4789` Mean-OOD
  pAUROC@20. Qwen3.5-4B became the lowest-cost observed frontier point.
- Ran a Qwen3.5-9B output-interface ablation. Immediate ordinal 0–10 scores
  reached only `0.4858` pAUROC@20, while rounding the saved binary
  probabilities to the same nominal resolution retained `0.6835`. Most of the
  loss therefore came from the generated-severity interface rather than score
  resolution alone.
- Prepared capacity and reasoning follow-ups for Qwen3.5-27B, GPT-OSS native
  reasoning, and Qwen visible-reasoning ordinal scoring. The first ordinal
  reasoning preflight was invalidated when a rationale reached its token cap;
  a corrected 4,096-token version was frozen but had not been evaluated at
  sign-off.
- Created the canonical cost-performance registry and reproducible SVG/PNG
  Pareto-frontier plot comparing Gleipnir baselines with the paper's published
  monitors.

### One-hour result-review extension

Reviewed and interpreted three completed autonomous benchmark runs:

- Qwen3.5-27B reached `0.8395` Mean-OOD pAUROC@20, producing a matched
  4B/9B/27B capacity curve of `0.6175`/`0.6976`/`0.8395`. The larger
  9B-to-27B gain established a strong capacity effect under the immediate
  binary-logit interface. At the paper's hosted-price proxy, the 27B result is
  a new observed frontier point, though its comparison with the paper's
  trained rationale-and-ordinal monitor is interface-level rather than a
  causal training comparison.
- GPT-OSS-120B with low native reasoning improved from `0.4789` to `0.5578`
  Mean-OOD pAUROC@20. The gain was heterogeneous across sources and did not
  close the gap to Qwen3.5-9B; at the canonical price, the cheaper and stronger
  Qwen3.5-4B baseline still dominates it.
- Qwen3.5-9B visible reasoning raised ordinal scoring from `0.4858` to `0.6656`
  Mean-OOD pAUROC@20, recovering 84.9% of the immediate ordinal deficit. It
  nevertheless remained `0.0320` below direct binary logits while costing
  8.7% more. This supports a partial-recovery interpretation: deliberation
  helps the generated severity interface, but direct decision-token logits
  remain the stronger deployment baseline.
- Checked the run audits and qualifications before accepting these results,
  including the frozen-date regeneration for GPT-OSS and bounded score-boundary
  recovery for 21 of 6,395 Qwen reasoned-ordinal rows.

The benchmark execution and result-record commits were autonomous; the one
hour counted here is for examining the completed evidence, checking the audit
details, and updating the research interpretation.

**Cumulative time:** 11 of 20 hours.

**Remaining:** 9 hours.

## 2026-08-31 — Session 3

**Time:** approximately 14:30–16:30, 2 hours

**Commit evidence:** `54e5473` through `51e30e9`, committed 15:08–16:32

Created a methods figure comparing Gleipnir's distillation pipeline with the
authors' approach, then began strict-OOD evaluation of the trained adapters:

- Built a reproducible SVG/PNG pipeline figure aligning the shared action-only
  input and deployable-monitor endpoint. It contrasts Sinha et al.'s filtered
  rationale-distillation stages—candidate generation, label-conditioned retry,
  external judging, and best-candidate selection—with Gleipnir's direct
  distillation of Kimi's binary decision-token distribution. Iteratively refined
  the branch structure, labels, arrows, and whitespace, with focused figure
  tests and a documented regeneration command.
- Froze the evaluation contract for the completed rank-128 soft-only adapters:
  six nested Qwen3.5-9B data-scaling conditions from 204 through 8,688 matched
  examples, plus separate 9B and 4B mixed-data conditions. All eight adapters
  will be scored on the same 6,395-row, six-source strict-OOD suite using the
  compact student prompt and normalized literal `0|1` decision-token logits.
- Implemented the evaluation preparation, causal-master and vLLM-serving parity
  canary, sequential one-H100 launcher, metric summarization, and artifact
  checks. Repaired launcher startup and normalized parity records before
  queueing the campaign. Full adapter results remain pending and have not yet
  been interpreted.

**Cumulative time:** 13 of 20 hours.

**Remaining:** 7 hours.
