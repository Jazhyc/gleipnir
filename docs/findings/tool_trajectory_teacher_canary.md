# Kimi K3 binary-logit teacher canary

Date: 2026-08-30. Status: completed bounded training-source canary. This is not
an OOD evaluation or a replication of the paper's final data manifest.

## Question and frozen design

The canary tests whether Kimi K3's literal next-decision distribution is already
a useful continuous monitor score when the model is asked to return only
`Prediction:0` or `Prediction:1`. A strong result would justify collecting
binary logit targets instead of visible teacher rationales; a weak result would
motivate restoring a deliberative output step or expanding the teacher policy.

The sample contains 100 rows, equally divided among the two source-label pairs
from each of `gemini_synthetic_1000.parquet` and
`unified_dataset_1000.parquet`. Rows come from the research-use-only
`Syghmon/blackboxmonitorsMATS` revision
`ccb77e37589baaee1f0e16ac9ab89ceac8f6cc8a`. We applied the paper's 28,000-token
input cutoff using `Qwen/Qwen3.5-9B` tokenizer revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, then selected deterministic
source-label-stratified prefixes under seed `20260830`. This left 1,817 of
2,000 candidate rows eligible.

The ignored input manifest has SHA-256
`8c15975aae9abf3f52acfcb475e05356ed3118dfa1d9133d2bd2b0d9856f8715`.
The teacher used prompt set `tool-trajectory-binary-v1`, template SHA-256
`2418cc55801deead8983d3bde6e35b59603f0080711ce8e12e1d7299487ac128`,
and an exact 5,029-character cache prefix with SHA-256
`d2d1678601ca82d78510d23f665ec097b75583bbc151ec2f727c4fda992bfda2`.
Requests used `moonshotai/kimi-k3`, provider-only Makora, temperature zero,
reasoning effort `none`, top-five logprobs, and no fallback.

## Result

| Population | N | AUROC | Stratified bootstrap 95% CI | Brier | Balanced accuracy | Recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pooled | 100 | **0.9746** | **0.9474--0.9944** | 0.0620 | 0.930 | 0.900 | 0.040 |
| Gemini synthetic | 50 | 0.9888 | 0.9632--1.0000 | 0.0474 | 0.960 | 1.000 | 0.080 |
| unified | 50 | 0.9880 | 0.9600--1.0000 | 0.0767 | 0.900 | 0.800 | 0.000 |

The unweighted mean source AUROC is `0.9884`. The generated binary choices have
AUROC and accuracy `0.9300`; using the relative `0|1` logits therefore adds
`0.0446` pooled AUROC over treating the one-token choice itself as the score.
There are 95 unique continuous scores among 100 rows. The positive-class score
median is `0.9669`, versus `0.0260` for the benign class.

This clears the intended sanity check: a single visible decision step does not
prevent Kimi K3 from producing a strong ranking signal on these upstream
synthetic training sources. The paper's roughly 0.95 frontier is an OOD
aggregate over different rows, prompts, teachers, and sometimes output spaces;
the present in-distribution result must not be presented as matching or beating
it. The bootstrap interval includes 0.95, but that is not evidence of metric
equivalence.

## Format and cache findings

The primary eight-token setting completed 97 rows. Two prompts repeatedly began
a prose explanation and exhausted the output budget before a terminal label;
one additional row exhausted retries on an upstream Makora 429. All three were
retried unchanged in a separate cache with `max_tokens=64` and produced an exact
binary completion. The final metrics include all 100 rows and retain both
request-setting hashes. Use a 64-token cap for future Kimi annotation even
though the requested interface remains one line; the cap is a format-recovery
allowance, not rationale supervision.

Provider prefix caching did **not** work for unique rows. The 97 primary
successes processed 1,086,912 prompt tokens with zero cache reads or writes.
The three recovery calls reported 7,680 cached of 14,800 prompt tokens because
those exact prompts had already been attempted; that is repeated-request
caching, not reuse of the shared teacher prefix. OpenRouter's endpoint metadata
reported `supports_implicit_caching=false` for Makora on Kimi K3, and the
explicit content-block breakpoint did not produce cross-row cache telemetry.
Do not begin the full teacher-cache campaign on the assumption that Makora will
reuse the policy prefix. Re-probe provider capabilities or choose a verified
cache-compatible route first.

Successful result rows report `$2.7740` total cost. This is a lower bound on
actual spend because the current generic CLI does not preserve usage from HTTP
200 responses that fail terminal-label parsing. Fix failure-attempt accounting
before scale-up; the teacher cache contract requires parse failures and their
token usage to remain auditable.

## Updated belief

Binary logit distillation remains the preferred first supervision arm. The
single-token boundary retains substantially more information than the sampled
hard decision and performs well enough to justify an OOD canary. The next
evidence should test the same frozen prompt on held-out source families, not
optimize this prompt on the present training rows. Full annotation remains
blocked by unverified cross-row input caching and by the need to preserve paid
parse-failure usage.
