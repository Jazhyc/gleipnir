# Matched Qwen 3.5 27B and Kimi K3 teacher agreement

2026-09-06. Complete diagnostic; not a held-out model-selection experiment.
Implementation: `experiments/teacher_agreement/`, initial commit `f2ab8ef`.

## Contract and validation

640 hash-selected unique full trajectories, 64 per source/label cell across
five training sources. Original Kimi rubric and exact rendered source-prompt
SHA match; Qwen uses its previously validated non-thinking binary-token
interface. Kimi scores are existing cached targets, with provider provenance
retained; no new paid calls or student training. Teacher chat/serving interfaces
and FP8 quantization are not held identical, so this is an operational teacher
comparison, not an isolated architecture experiment.

Qwen/Qwen3.5-27B-FP8 revision
`97f5941bf617e31c5e237364a8602ce3f03a551a`, TP2 on both H100 SXM5 GPUs,
Torch 2.11.0 / vLLM 0.24.0, prefix caching disabled. 7,029,373 prompt tokens,
maximum 30,149; initialization 52.63 seconds. Balanced-plus-longest canary
passed batch/singleton comparison: MAE 0.002924, maximum 0.014350. Canary
records retain initial batched scores; subsequent scoring uses singleton
requests. This is not an FP8/BF16 parity test.

All 640 unique outputs were pulled and re-analyzed locally with strict
identity, label, prompt-hash, config-hash and raw-score checks.

## Results

Positive probability denotes problematic behavior. Compare probabilities
normalized over literal 0/1, and margins `logprob(1) - logprob(0)`; absolute
vocabulary logprobs are not directly comparable across model tokenizers.

| Statistic | Result |
| --- | ---: |
| Probability MAE | 0.098254 |
| Mean Qwen minus Kimi probability | -0.078878 |
| Spearman agreement | 0.925327 |
| Threshold disagreements | 56/640 (8.75%) |
| Kimi positive / Qwen negative | 53 |
| Qwen positive / Kimi negative | 3 |
| Qwen / Kimi mean max(p, 1-p) | 0.962203 / 0.939829 |
| Qwen / Kimi Brier against labels | 0.116556 / 0.071498 |

Descriptive least-squares margin relation: Qwen margin approximately
`1.191083 * Kimi margin - 0.782365`. This is a description, not a proposed
calibration or evidence that an affine transformation resolves disagreements.

| Source (128 each) | Probability MAE | Qwen−Kimi bias | Threshold disagreement |
| --- | ---: | ---: | ---: |
| Bash Arena | 0.152422 | -0.117663 | 13.28% |
| Bash Bench | 0.039939 | -0.033961 | 3.13% |
| CoT Red Handed | 0.051674 | -0.022736 | 2.34% |
| Gloom | 0.174749 | -0.162407 | 18.75% |
| STRIDE | 0.072487 | -0.057621 | 6.25% |

Gloom Brier is 0.166541 for Qwen versus 0.035058 for Kimi. Negative Qwen
bias also appears within every Kimi provider stratum (Fireworks -0.083570,
Makora -0.073447, Morph -0.099102), although provider groups are not randomized
and Morph has only 25 sampled rows.

## Interpretation

There is meaningful teacher-target mismatch beyond the measured startup
numerical variation. Qwen is more confident overall while assigning less
probability to problematic behavior; disagreements are strongly directional
and source-dependent. Rank agreement is high but imperfect, so this is not
merely a global temperature difference. On this balanced training diagnostic,
Kimi also has better Brier loss overall (not on every source).

This supports mixed-teacher conflict as a plausible contributor to the weak
prefix-supervision result, but does not establish causality. Prefix visibility,
the changed prefix rubric, cache execution-history noise, sampling and loss
weighting remain separate possible contributors. No rescaling, new training or
promotion follows automatically. The balanced sample is not the original
source-weighted population, and no uncertainty interval was estimated.

## Artifacts

### Label-based calibration follow-up (2026-09-07)

Same 640 paired full-boundary scores, unchanged targets and existing labels.
Positive-class ECE weights each equal-width bin's absolute difference between
mean probability and observed positive fraction by its sample count. Empty
bins contribute zero. Brier and natural-log loss are proper scoring rules,
not pure measures of calibration; log loss clips only numerically at 1e-15.

| Diagnostic | Qwen 3.5 27B FP8 | Kimi K3 |
| --- | ---: | ---: |
| ECE, 10 equal-width bins | 0.117087 | 0.044054 |
| ECE, 5 / 20 bins | 0.110206 / 0.122986 | 0.035154 / 0.048228 |
| Brier | 0.116556 | 0.071498 |
| Log loss (nats) | 0.522498 | 0.245876 |
| Accuracy at 0.5 | 0.857813 | 0.910938 |
| Mean predicted-class confidence | 0.962203 | 0.939829 |
| Mean confidence minus accuracy | 0.104391 | 0.028891 |
| Mean positive probability minus prevalence | -0.104908 | -0.026031 |

Both teachers are overconfident on average on this sample, more markedly Qwen.
Kimi has lower estimated calibration error across all three bin counts. This
is not evidence that bimodality itself is a defect, nor that a fitted correction
would improve transfer. The observed label prevalence is 0.5 by construction;
source/label-balanced sampling alters calibration relative to other deployment
mixtures. Labels may be imperfect measures of rubric-defined behavior. No
calibrator is fitted and no new teacher or student run is launched.

Reliability diagrams label bin counts and show descriptive 95% Wilson intervals
for observed rates (independent-Bernoulli assumption, no grouped uncertainty).
Sparse middle bins are correspondingly uncertain. Exact numerical tables,
including per-source diagnostics, live in `results/teacher_agreement/calibration.json`.
Input paired-score SHA256:
`126d12f58889603478941b99faa5f2c581a57ffd43e255849e461dde81b85e07`.

### Original agreement artifacts

`results/teacher_agreement/agreement.json` contains pooled, source and provider
statistics; `paired_scores.jsonl` contains the matched probability/margin pairs.
Original Kimi logprobs remain in `kimi.jsonl`; raw Qwen logprobs remain in
`qwen/predictions.jsonl`.

- Frozen benchmark SHA: `15d86e10a0c9feeae82fd4d7a08afd066ae3ad9014e2c1eff5aa079c338f1054`
- Matched Kimi SHA: `ecc9c6a9211cefee051d562eb58ebcb2f7bd8e74e00394039a20d43b6030a0bc`
- Qwen predictions SHA: `6b09a5ddbcbc0dc946c25b25926043549885cf3cac8e490392fcce6c9a4db7e5`
