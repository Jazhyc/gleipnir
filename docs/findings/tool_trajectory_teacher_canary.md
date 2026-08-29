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
a prose explanation and used all eight allowed tokens before reaching a terminal
label; they did not exceed the configured limit. One additional row exhausted
retries on an upstream Makora 429. All three were retried unchanged in a
separate cache with `max_tokens=64` and produced an exact binary completion. The
final metrics include all 100 rows and retain both request-setting hashes. This
recovery established that a larger unconstrained allowance can hide formatting
failures, but it is not the preferred scaled interface.

Provider prefix caching did **not** work for unique rows under the canary route.
The 97 primary successes processed 1,086,912 prompt tokens with zero cache reads
or writes. The three recovery calls reported 7,680 cached of 14,800 prompt
tokens because those exact prompts had already been attempted; that is
repeated-request caching, not reuse of the shared teacher prefix.

Follow-up transport probes isolated two causes. First, manually pinning
`provider.only=["Makora"]` produced zero reuse even for two prompts with about
5,400 shared leading tokens. OpenRouter documents that manual provider ordering
disables its sticky-routing mechanism; the `provider.only` probe behaved the
same way. With the provider pin removed,
`provider.sort="price"`, a fixed `session_id`, and harmless public test text,
OpenRouter still selected Makora and the second 10,574-token request read 9,216
tokens from cache. Its reported cost fell from `$0.02677` cold to `$0.00584`
warm. Second, caching has a size threshold: a 1,287-token prompt produced no
reuse, while a 2,447-token prompt produced a 1,536-token cache read. Observed
reads are multiples of 512 tokens. Treat roughly 2,000 invariant tokens as the
observed minimum until a provider contract states otherwise. The present
5,029-character teacher prefix is too short: an entire small teacher request,
including its trajectory, was only 1,109 provider-counted prompt tokens. This
is a transport limitation, not a reason to lengthen the policy. Even the
optimistic upper bound from caching 1,109 fixed tokens is about `$0.00254` per
warm request at the probed Makora rates; padding could change model behavior and
consume context while addressing only a minority of the roughly 11,200-token
average canary request.

The preferred route can remain fail-closed on Makora's lower price without a
provider pin. A fresh harmless probe set `provider.max_price` to `$2.55/M`
prompt and `$12.75/M` completion, admitted only Makora at current endpoint
prices, and retained sticky caching: the first 2,403-token request was cold and
the second read 1,536 tokens from cache. If Makora raises its price or becomes
unavailable, this ceiling makes the request fail instead of silently using a
costlier provider. Re-query endpoint prices before each campaign rather than
assuming these constants remain current.

OpenRouter's Kimi K3 endpoint and Makora also support strict structured output
and token logprobs together. A strict scalar JSON Schema with root integer enum
`[0, 1]` and `max_tokens=1` returned the single content token `0`; its top-five
alternatives included both literal `0` and `1`. A later paired canary showed
that the strict decoder materially changes the relative binary margin even when
the scalar prompt is identical. It is therefore supported but is not a
transparent raw-logit replacement. See
`docs/findings/tool_trajectory_structured_output_canary.md`. A JSON-object
schema also worked, but consumed six completion tokens and is unnecessary.

Successful result rows report `$2.7740` total cost. This is a lower bound on
actual spend because the current generic CLI does not preserve usage from HTTP
200 responses that fail terminal-label parsing. Fix failure-attempt accounting
before scale-up; the teacher cache contract requires parse failures and their
token usage to remain auditable.

## Updated belief

Binary logit distillation remains the preferred first supervision arm. The
single-token boundary retains substantially more information than the sampled
hard decision and performs well enough to justify an OOD canary. Preserve the
successful teacher policy rather than padding it for a cache threshold. Before
scaling, prefer the tested prompt-only scalar interface over strict structured
decoding, remove manual provider pinning in favor of a Makora-rate price ceiling
and stable session ID, and accept that distinct rows will not share-cache under
the present short policy. Exact retries can still benefit. The prompt-only
scalar interface needs a larger format-adherence canary before full annotation,
which also requires paid parse-failure usage to be preserved.
