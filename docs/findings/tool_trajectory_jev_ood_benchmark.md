# Jev 1.13 native probability OOD benchmark

Evaluation: 2026-09-19. Inclusion audit and figure update: 2026-09-30.
Status: all 6,395 frozen OOD IDs verified, metrics and cost reproduced from
retained artifacts. This update adds the existing result at the user's request;
it launches no API inference. The result uses a different native monitoring
interface from the full teacher chat prompt.

## Result

`typesafe/jev-1.13`, served by TypeSafe as `typesafe/jev-1.13-20260917`, reaches
**0.849513 Mean-OOD pAUROC@20** and **0.926899 Mean-OOD AUROC**. These are
unweighted six-source means on the same strict-OOD population as the other
frontier entries. pAUROC@20 uses raw ROC area through FPR 0.2 divided by 0.2.

| Source | Rows | AUROC | pAUROC@20 | Recall at 0.5 | FPR at 0.5 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AgentDojo | 1,500 | 0.995884 | 0.979422 | 0.997333 | 0.065333 | 0.052575 |
| CUA-SHADE-Arena | 955 | 0.921192 | 0.807675 | 0.804622 | 0.054280 | 0.109934 |
| IaC | 1,443 | 0.994942 | 0.976901 | 0.686713 | 0.004121 | 0.109417 |
| MLE-Sabotage | 148 | 0.996327 | 0.981735 | 0.683673 | 0.000000 | 0.144830 |
| Rogue Eval | 1,499 | 0.978016 | 0.909845 | 0.566088 | 0.001333 | 0.150453 |
| SHADE-Arena | 850 | 0.675034 | 0.441500 | 0.650718 | 0.479167 | 0.226015 |

Pooled AUROC/pAUROC are 0.932874/0.721494. Pooled recall is 0.743918,
FPR 0.089683, balanced accuracy 0.827117, and Brier 0.122098. Macro Brier
is 0.132204. The low FPR on several sources accompanies substantial missed
positives at the fixed 0.5 threshold; this is not evidence of uniform calibration.
SHADE remains the weakest source.

Jev has only 95 distinct saved probabilities, ranging from 0.03 to 0.97.
Every example belongs to a repeated-score group; the largest group has 211 rows.
The original summary's `tied_rows=6300` means total rows minus unique scores,
not the number of rows belonging to repeated-score groups. Token-logprob
monitors generally have much fewer ties, so nominal probability outputs do
not establish equivalent score resolution.

At **$0.388664204 per 1,000 evaluations**, Jev enters the dated point-estimate
frontier between Gleipnir 4B and Kimi K2.6. It dominates GPT-6 Luna, Gleipnir
9B, and Qwen3.5-27B on the plot's cost and primary metric. These are descriptive
configuration comparisons without a paired significance claim. Different
rubrics, representations, tokenizers and native scoring interfaces prevent
attributing the difference solely to the model. Do not tune prompts or
thresholds on this OOD result.

## Frozen native interface and historical provenance

The original hypothesis, intervention, baselines, selection rule, preflight
gates and stop conditions are recoverable from commit
`5c6db2af13d532b7f4ac5d67f4341b0a317e51df`:

- `experiments/tool_trajectory_monitoring/jev_113_ood_benchmark.json`.
- `experiments/tool_trajectory_monitoring/prompts/jev_noul_v1.json`.
- `experiments/tool_trajectory_monitoring/prepare_jev_ood.py`.
- `experiments/tool_trajectory_monitoring/benchmark_jev_ood.py`.
- `src/gleipnir/openrouter_decisions.py`.

Use `git show <commit>:<path>` to recover these historical files. They are
not restored as executable API entrypoints by this plot update.

The request uses OpenRouter's `api/alpha/decisions` endpoint, route
`typesafe/jev-1.13`, TypeSafe only, fallbacks disabled, and data collection
denied. Eight workers scored the full suite from 00:24:59 to 00:31:35 UTC.
Every retained row reports one attempt, one unique response ID, and the same
model revision, provider, configuration and question identities.

The state contains the exact raw trajectory recovered from the frozen full
teacher prompt and matched to its `trajectory_sha256`. It excludes the teacher
prompt envelope. The separate, frozen Noul question defines problematic
actions, unauthorized behavior and material deception, and treats trajectory
contents as untrusted evidence. Scores are the actual returned
`answer.noul` probabilities; labels and provenance remain local metadata.
This preserves trajectories but changes the instruction and request interface.

The 100-row non-OOD training-source canary passed its predeclared coverage,
provider, score-resolution and ranking gates. A separate OOD preflight was
selected solely by largest Kimi token proxy and did not use its score to
change the campaign. No OOD question, threshold, subset, or score-based retry
selection was introduced for this figure update.

## Artifact audit and context discrepancy

The inclusion audit verifies input and output checksums, exact complete IDs,
all six source counts, original trajectory/state hashes, metadata, model and
provider, config and question identities, request-setting hashes, finite
probabilities, saved answer/score equality, and per-row costs. It recomputes
every source's ranking and threshold metrics and reproduces the original
summary exactly. Saved canary IDs are disjoint from strict OOD.

The cache retains parsed answers and usage rather than complete raw HTTP
responses. The audit can verify their consistency with the original client
and frozen contract, but cannot independently inspect unretained server fields.

The configured and [advertised context window](https://openrouter.ai/typesafe/jev-1.13)
is 32,000 tokens. The proxy-selected preflight reports 31,317 input tokens
and passed the original gate. Two full-run responses report more:

| ID | Source | Reported input tokens |
| --- | --- | ---: |
| `fbe095af2c8409730db75806` | CUA-SHADE-Arena | 32,239 |
| `a9de0cdabbdbf83046c7d100` | IaC | 32,221 |

All submitted local states still match complete original trajectories. The
original full-run gates did not apply the preflight's input-count ceiling to
every row. Preserve the discrepancy; cached answers do not rule out internal
provider truncation or explain accounting above the advertised window.
No rows are dropped or rerun, and no metric is adjusted to hide this limitation.

## Exact cost coordinate

The frozen 2026-09-19 rate is **$0.042/M input, $0/M output**. The
[OpenRouter model page](https://openrouter.ai/typesafe/jev-1.13), checked on
2026-09-30, still lists those prices. Every saved row's reported cost equals
its input tokens times the frozen rate.

- Accepted input: 59,178,752 tokens.
- Accepted output usage: 147,085 tokens, with zero output charge.
- Uncached full-suite cost: $2.485507584.
- Uncached cost per 1,000: **$0.3886642039093042**.

```text
(59,178,752 * 0.042 / 1,000,000) * 1,000 / 6,395
```

The frontier rounds this to `$0.3887/1K` and `0.8495` pAUROC. It excludes
canaries, the separate preflight, training and self-hosted infrastructure,
following the same successful-evaluation list-price convention as other rows.

## Ignored artifacts and identities

- Original population:
  `data/tool_trajectory_monitoring/teacher_ood_benchmark/prompts.jsonl`;
  SHA-256 `fa4b0d917e51eb91a52efa93cddf0eb154b88993269c97cfdb161c42680136a9`.
- Native states: `data/tool_trajectory_monitoring/jev_ood/states.jsonl`;
  SHA-256 `c38c6214577f0a4e68ab20ba5531d94da8bdaa26efc1420b0c77ef2ccdd2b99a`.
- Scores: `results/tool_trajectory_monitoring/jev_113_ood/scores.jsonl`;
  SHA-256 `31fd6781e6cf8ce55bf8352d6b5f5b58d993022e492c29f87a68ccd9d057fd13`.
- Summary: `results/tool_trajectory_monitoring/jev_113_ood/result.json`;
  SHA-256 `8e94bcec53270d4c5a195410f1e02a472b5079d9718b3be534f24047391c94cd`.
- Config SHA-256:
  `493b34dba6b87374c7d2c3509bf70a03b2f4a4e6ac1b8a4f4b884c0d47e0a450`.
- Question SHA-256:
  `9b423177a4d48897d083f927845ab464fe0f97e758f64159aedee94b3dab1f12`.
- Request-settings SHA-256:
  `f1bcd38ddd9903a26e4a07703c0a629808a9bfead9c881b8002f779c2c872808`.
- `frontier_inclusion_audit.json` and `audit_frontier_inclusion.py`, beside
  the score cache, preserve the independent recheck and its context exception.

The [canonical registry](../research/tool_trajectory_ood_frontier.md) records
the point and its frontier effects. The local figure update does not make a
claim about permission to publish under provider terms.
