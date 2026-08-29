# Kimi K3 scalar structured-output canary

Date: 2026-08-30. Status: completed paired 20-row training-source canary. The
strict structured condition did **not** meet the frozen logit-parity rule.

## Question and design

This canary asks two separate questions before scaling teacher annotation:

1. Does changing only the teacher output interface from `Prediction:0|1` to a
   single JSON integer preserve the useful ranking of the original teacher?
2. Under that same scalar-only prompt, does Makora's strict JSON Schema preserve
   the unconstrained `0|1` logit target?

The input uses the first 20 balanced, interleaved rows from the frozen 100-row
teacher canary: five examples from every source-label stratum. The base input
SHA-256 is
`8c15975aae9abf3f52acfcb475e05356ed3118dfa1d9133d2bd2b0d9856f8715`;
the materialized paired input SHA-256 is
`a142a39802d86b25a74b3a7eefa0e060b3134eefe24dd4f1dd1abb401da39461`.
The monitoring policy is unchanged. Only its final output-interface paragraph
is replaced by the explicit candidate in
`experiments/tool_trajectory_monitoring/prompts/teacher_scalar_interface.txt`,
SHA-256
`b4893bf3b65fb3e32d70cd54d5d0b7d9d6bb850fe64bb5f9f6e2c38169ff120b`.

The three paired conditions are:

- the existing cached `Prediction:0|1` reference;
- the candidate scalar-only interface, unconstrained, with `max_tokens=1`;
- the identical candidate interface with strict root-integer JSON Schema enum
  `[0, 1]` and `max_tokens=1`.

All new calls used `moonshotai/kimi-k3`, Makora's current price ceiling
(`$2.55/M` prompt and `$12.75/M` completion), no fallback, temperature zero,
reasoning disabled, and top-five output logprobs. An attempted user-side
`Prediction:` suffix and an assistant-prefill probe both caused Kimi to start a
new answer with the token `Prediction`; OpenRouter/Makora therefore cannot
expose the original assistant-side label boundary through continuation.

Before paid calls, strict-versus-unconstrained acceptance required all 20 pairs
from Makora, Pearson and Spearman correlations at least `0.999`, maximum absolute
score change at most `0.01`, AUROC change at most `0.01`, and no threshold
flips. Practical comparison to the old interface additionally required
structured Spearman at least `0.95` and no more than `0.05` AUROC degradation.

## Result

| Condition | AUROC | Hard accuracy | Unique scores |
| --- | ---: | ---: | ---: |
| original `Prediction:0|1` reference | 0.960 | 0.900 | 19/20 |
| scalar prompt, unconstrained | **0.970** | 0.900 | 20/20 |
| scalar prompt, strict schema | 0.960 | 0.900 | 19/20 |

Per-source AUROC was `1.00` for every condition on Gemini synthetic. On unified,
the reference and unconstrained scalar condition scored `1.00`, while strict
schema scored `0.96`. These ten-row source estimates are descriptive only.

Strict decoding preserved hard decisions relative to the unconstrained scalar
condition: there were zero generated-label disagreements and zero `0.5`
threshold flips. It did **not** preserve the soft target:

| Strict schema versus unconstrained scalar | Value |
| --- | ---: |
| Pearson | 0.9860 |
| Spearman | 0.9816 |
| mean absolute score change | 0.0440 |
| maximum absolute score change | 0.1662 |
| mean absolute binary-margin change | 0.8438 |
| maximum absolute binary-margin change | 2.1250 |

The strict condition therefore failed the frozen parity rule. Its practical
ranking remained strong: versus the original reference, strict-schema Spearman
was `0.9823` and Pearson was `0.9780`. The unconstrained scalar interface was
slightly closer to the original, with Spearman `0.9853`, Pearson `0.9886`, and
mean absolute score change `0.0404` rather than `0.0530`.

Both new conditions returned a valid one-token scalar for all 20 rows. Every
successful call was served by Makora. The unconstrained successful rows report
`$0.56509`; the strict rows report `$0.10995` because running second reused
201,984 of 225,161 prompt tokens (`89.7%`) from exact prompts. These `$0.67504`
stored-success costs exclude earlier paid HTTP-200 format probes because the
generic client still does not preserve failed-response usage. Six strict calls
hit Makora shared-pool 429s at concurrency four and completed when resumed
serially.

## Updated belief

Do not treat Makora structured decoding as a transparent way to obtain the raw
teacher distribution. It preserves ranking and decisions here, but materially
changes the binary margin. Never mix strict and unconstrained teacher targets in
one training arm.

The candidate prompt-only scalar interface is the better primary route for a
raw-logit campaign: it produced 20/20 valid one-token responses, was marginally
closer to the original teacher, and had the best AUROC in this small sample.
Before a full annotation campaign, test its format adherence on the remaining
80 frozen canary rows and preserve paid parse-failure usage. Strict schema
remains viable only as a separately named supervision condition whose altered
calibration is accepted intentionally, not as a per-row recovery fallback.

The ignored manifest, paired score caches, and full JSON summary are under
`data/tool_trajectory_monitoring/structured_output_canary/` and
`results/tool_trajectory_monitoring/` and can be reconstructed with:

```bash
python -m experiments.tool_trajectory_monitoring.structured_output_canary prepare
gleipnir-openrouter --input data/tool_trajectory_monitoring/structured_output_canary/prompts.jsonl --output results/tool_trajectory_monitoring/kimi_k3_scalar_unstructured_canary.jsonl --model moonshotai/kimi-k3 --binary-output-mode scalar --max-tokens 1 --provider-max-prompt-price 2.55 --provider-max-completion-price 12.75 --no-allow-fallbacks
gleipnir-openrouter --input data/tool_trajectory_monitoring/structured_output_canary/prompts.jsonl --output results/tool_trajectory_monitoring/kimi_k3_scalar_structured_canary.jsonl --model moonshotai/kimi-k3 --binary-output-mode scalar --structured-output --max-tokens 1 --provider-max-prompt-price 2.55 --provider-max-completion-price 12.75 --no-allow-fallbacks
python -m experiments.tool_trajectory_monitoring.structured_output_canary compare
```
