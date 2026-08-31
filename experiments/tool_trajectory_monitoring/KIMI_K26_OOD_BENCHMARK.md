# Kimi K2.6 Inceptron binary-logit OOD benchmark

Status: completed on all 6,395 strict-OOD rows on 2026-09-01.

## Hypothesis and design

This benchmark tests whether unadapted `moonshotai/Kimi-K2.6`, evaluated under
Gleipnir's frozen full teacher specification and immediate literal `0|1`
logits, exceeds the paper's `0.841` Mean-OOD pAUROC@20 Kimi-K2.6 SFT point at
lower uncached hosted inference cost. The intervention uses OpenRouter's
Inceptron endpoint with reasoning disabled instead of the paper's trained short
prompt and generated rationale plus 0--10 score.

The evaluation reuses all 6,395 unchanged strict-OOD rows and the byte-identical
teacher prompts used for the completed Kimi K3 benchmark. No OOD label may
influence the prompt, provider, settings, score extraction, threshold, subset,
or stopping decision. The executable design and failure gates are frozen in
[`kimi_k26_ood_benchmark.json`](kimi_k26_ood_benchmark.json).

## Cost and interface preflight

The pinned public K2.6 tokenizer counts 53,145,980 rendered non-thinking chat
input tokens, or 8,310.552 per row, with a maximum of 27,082. At Inceptron's
2026-08-31 OpenRouter rates of `$0.53/M` input and `$3.39/M` output, ten
completion tokens per row project to `$28.3842` for the suite, or `$4.4385` per
1,000 evaluations. The fail-closed campaign ceiling is `$33.00`; cache discounts
are excluded from the primary cost coordinate.

Before the full run, the first 12 rows of the existing interleaved OOD canary
must cover every source-label stratum and pass the exact terminal-format,
provider-identity, non-thinking, price, and two-literal-logprob gates in the
frozen config. Inceptron is pinned with fallbacks disabled and OpenRouter must
enforce every requested parameter.

After that technical gate, score all 600 rows of the already-frozen OOD canary,
which contains 50 examples from each source-label stratum. Report its raw
normalized Mean-OOD pAUROC@20 as a plausibility check before the full result.
The 600-row score is descriptive: it must not change the prompt, provider,
request settings, threshold, full-suite membership, or launch decision.

The technical gate passed model, provider, price, reasoning, completion, and
two-literal-logprob checks. Inceptron consistently prepended one ASCII space to
the otherwise exact `Prediction:0|1` line. The existing terminal parser already
accepts surrounding whitespace. This label-blind provider-format amendment was
documented before extending the cache; no prompt, request setting, score
boundary, or parser code changed.

The completed 600-row quality canary reached `0.936750` Mean-OOD pAUROC@20 and
`0.978933` Mean-OOD AUROC. Pooled pAUROC@20/AUROC were `0.910889/0.973306`.
All 600 rows were served by Inceptron with zero reasoning tokens and both
literal decision logprobs. Reported spend was `$2.465338`, including a 21.67%
provider cache-read fraction. The canary is encouraging but is not a frontier
point and does not replace the frozen 6,395-row evaluation.

## Definitive result

The complete run reaches `0.919404` Mean-OOD pAUROC@20 and `0.975107`
Mean-OOD AUROC. It exceeds Kimi K3 by `0.011038` pAUROC@20 and the paper's
Kimi-K2.6 SFT point by `0.078404`. The source-balanced gain over K3 is driven
by SHADE-Arena: K2.6 improves that source by `0.150168` while declining on the
other four non-tied sources.

All 6,395 rows have exactly one score from Inceptron's dated
`moonshotai/kimi-k2.6-20260420` endpoint, both literal-token logprobs, and zero
reasoning tokens. Exact observed usage is 53,145,980 prompt tokens and 25,580
completion tokens. At the frozen uncached list rates, the suite costs
`$28.254086`, or `$4.418153/1K`; reported cache-discounted successful-row spend
was `$22.930851`.

The detailed per-source, cost, retry, tie, coverage, and artifact audit is in
[`tool_trajectory_kimi_k26_ood_benchmark.md`](../../docs/findings/tool_trajectory_kimi_k26_ood_benchmark.md).

## Interpretation contract

A completed point above `0.839474` enters the observed open-model
cost--performance frontier. A point at least `0.841` dominates the paper's
Kimi-K2.6 SFT point on the registry's point-estimate cost convention. Only a
point above the completed Kimi K3 binary-logit result of `0.9084` dominates the
current high-quality open-model endpoint. Because prompt and score interface
differ from the paper's SFT condition, any positive result is evidence for
interface-level cost--performance dominance, not a matched causal rejection of
SFT.

The completed point satisfies all three predeclared thresholds: it enters the
frontier, dominates the paper's K2.6 SFT point, and exceeds the prior Kimi K3
open-model endpoint.
