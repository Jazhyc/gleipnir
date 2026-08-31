# Tool-trajectory OOD cost--performance frontier

Status: living registry, established 2026-08-30.

This file is the canonical comparison table for completed evaluations on the
six-source, 6,395-row strict OOD suite from Sinha et al., *Training Deliberative
Monitors for Black-Box Scheming Detection*. Add every future Gleipnir OOD
evaluation here once coverage and cost accounting pass the gates below. Do not
use canaries, incomplete runs, or tuned-on-OOD variants as frontier points.

The current plot is tracked as a sharp vector
[`SVG`](../../figures/tool_trajectory_ood_frontier.svg) with a
[`PNG`](../../figures/tool_trajectory_ood_frontier.png) fallback and is
regenerated directly from the table below with
`python scripts/plot_tool_trajectory_ood_frontier.py`. Shared plotting helpers
live in `src/gleipnir/plotting.py`; see
[`../../figures/README.md`](../../figures/README.md) before adding another plot.

## Comparison contract

The performance coordinate is the paper's raw normalized Mean-OOD pAUROC@20:

```text
pAUROC@20(D) = (1 / 0.2) * integral from FPR=0 to 0.2 of TPR(FPR) dFPR
Mean-OOD     = unweighted mean of pAUROC@20(D) over the six OOD datasets
```

This is not scikit-learn's standardized partial AUC. The cost coordinate is
uncached list-price USD per 1,000 evaluations:

```text
cost_per_1k = (mean_input_tokens * input_USD_per_Mtok
               + mean_output_tokens * output_USD_per_Mtok) / 1000
```

Provider-billed reasoning tokens count as output. Prompt caching, batch
discounts, failed-request charges, training cost, and self-hosted total cost of
ownership are excluded from the primary coordinate. Record them separately.

The paper rows below are the published three-seed means rounded to three decimal
places. Gleipnir's Qwen rows are deterministic temperature-zero evaluations;
the GPT-OSS low-reasoning row uses one frozen sampling seed before the
binary-logit boundary. They are comparable in dataset, metric, and uncached
cost convention, but do not reproduce the paper's trained interfaces.

## Reference table

Rows are ordered by cost. `Frontier` is point-estimate nondominance after adding
all Gleipnir rows currently in this file: no other row is both no more expensive
and at least as accurate, with one strict improvement.

| Origin | Monitor | Evaluation interface | Uncached inference cost (USD / 1,000 evaluations) | Mean-OOD pAUROC@20 | Frontier |
| --- | --- | --- | ---: | ---: | :---: |
| **Gleipnir** | **Qwen3.5-4B base** | **Full binary specification, immediate `0\|1` logits** | **0.2842** | **0.6175** | **yes** |
| **Gleipnir** | **GPT-OSS-120B base** | **Full binary specification, immediate `0\|1` logits** | **0.3270** | **0.4789** | **no** |
| **Gleipnir** | **GPT-OSS-120B base** | **Full binary specification, low analysis then `0\|1` logits** | **0.3442** | **0.5578** | **no** |
| Paper | GPT-OSS-20B SFT | Trained, short prompt, generated rationale + 0--10 score | 0.347 | 0.656 | yes |
| Paper | Qwen3.5-4B SFT+RL | Trained, short prompt, generated rationale + 0--10 score | 0.358 | 0.684 | yes |
| Paper | GPT-OSS-120B SFT | Trained, short prompt, generated rationale + 0--10 score | 0.511 | 0.718 | yes |
| Paper | Nemotron-30B SFT | Trained, short prompt, generated rationale + 0--10 score | 0.590 | 0.578 | no |
| Paper | Qwen3-8B SFT+RL | Trained, short prompt, generated rationale + 0--10 score | 0.748 | 0.628 | no |
| Paper | Qwen3-32B SFT+RL | Trained, short prompt, generated rationale + 0--10 score | 0.869 | 0.703 | no |
| **Gleipnir** | **Qwen3.5-9B base** | **Full binary specification, immediate `0\|1` logits** | **0.9468** | **0.6976** | **no** |
| **Gleipnir** | **Qwen3.5-9B base (ordinal)** | **Full severity specification, immediate 0--10 integer** | **0.9628** | **0.4858** | **no** |
| **Gleipnir** | **Qwen3.5-9B base (reasoned binary)** | **Full binary specification, visible analysis then `0\|1` logits** | **1.0054** | **0.6923** | **no** |
| **Gleipnir** | **Qwen3.5-9B base (reasoned ordinal)** | **Full severity specification, visible analysis then 0--10 integer** | **1.0293** | **0.6656** | **no** |
| Paper | Nemotron-120B SFT | Trained, short prompt, generated rationale + 0--10 score | 1.251 | 0.780 | yes |
| **Gleipnir** | **Qwen3.5-27B base** | **Full binary specification, immediate `0\|1` logits** | **1.8476** | **0.8395** | **yes** |
| Paper | Qwen3.5-27B SFT+RL | Trained, short prompt, generated rationale + 0--10 score | 2.775 | 0.831 | no |
| Paper | GPT-5.4 nano prompted | Full specification, generated rationale + 0--10 score | 3.387 | 0.752 | no |
| Paper | Gemini 3.1 Flash-Lite prompted | Full specification, generated rationale + 0--10 score | 4.157 | 0.815 | no |
| Paper | Kimi-K2.6 SFT | Trained, short prompt, generated rationale + 0--10 score | 10.175 | 0.841 | yes |
| Paper | Claude Haiku 4.5 prompted | Full specification, generated rationale + 0--10 score | 19.746 | 0.794 | no |
| **Gleipnir** | **Kimi K3 binary logits** | **Full binary specification, immediate `0\|1` logits** | **25.1177** | **0.9084** | **yes** |
| Paper | GPT-5.4 prompted | Full specification, generated rationale + 0--10 score | 44.376 | 0.899 | no |
| Paper | Gemini 2.5 Pro prompted | Full specification, generated rationale + 0--10 score | 45.254 | 0.802 | no |
| Paper | Claude Sonnet 4.6 prompted | Full specification, generated rationale + 0--10 score | 58.850 | 0.929 | yes |
| Paper | Gemini 3.1 Pro prompted | Full specification, generated rationale + 0--10 score | 73.218 | 0.940 | yes |
| Paper | Claude Opus 4.6 prompted | Full specification, generated rationale + 0--10 score | 94.794 | 0.942 | yes |

The 17 paper reference rows reproduce Appendix E, Table 4. Their exact token
counts and prices remain in the paper rather than being duplicated here. The
Gleipnir rows are backed by the completed
[`tool_trajectory_gpt_oss_ood_benchmark.md`](../findings/tool_trajectory_gpt_oss_ood_benchmark.md),
[`tool_trajectory_gpt_oss_low_reasoning_ood_benchmark.md`](../findings/tool_trajectory_gpt_oss_low_reasoning_ood_benchmark.md),
[`tool_trajectory_qwen4b_ood_benchmark.md`](../findings/tool_trajectory_qwen4b_ood_benchmark.md),
[`tool_trajectory_qwen_ood_benchmark.md`](../findings/tool_trajectory_qwen_ood_benchmark.md),
[`tool_trajectory_qwen27b_ood_benchmark.md`](../findings/tool_trajectory_qwen27b_ood_benchmark.md),
[`tool_trajectory_qwen_ordinal_ood_benchmark.md`](../findings/tool_trajectory_qwen_ordinal_ood_benchmark.md),
[`tool_trajectory_qwen_binary_reasoning_ood_benchmark.md`](../findings/tool_trajectory_qwen_binary_reasoning_ood_benchmark.md),
[`tool_trajectory_qwen_ordinal_reasoning_ood_benchmark.md`](../findings/tool_trajectory_qwen_ordinal_reasoning_ood_benchmark.md)
and
[`tool_trajectory_kimi_ood_benchmark.md`](../findings/tool_trajectory_kimi_ood_benchmark.md)
audits and the exact accounting in
[`tool_trajectory_inference_economics.md`](tool_trajectory_inference_economics.md).

## Qwen3.5-4B base result

The unadapted Qwen3.5-4B full-teacher-prompt baseline reaches `0.6175`
Mean-OOD pAUROC@20 and `0.8549` Mean-OOD AUROC. Its exact 9,466.7113-token
mean input and one-token output cost `$0.2842/1K` at the paper's hosted proxy
of `$0.03/M` input and `$0.15/M` output. OpenRouter did not list a Qwen3.5-4B
route on 2026-08-30, so this coordinate is explicitly an Artificial Analysis
proxy carried forward from the paper, not a current OpenRouter quote.

The 4B base becomes the lowest-cost observed frontier point and dominates the
GPT-OSS-120B base, improving pAUROC@20 by `0.1386` while costing `13.1%` less.
It does not dominate the next paper point: GPT-OSS-20B SFT costs `$0.0628/1K`
more and improves pAUROC@20 by `0.0385`. The matched 9B base gains `0.0801`
pAUROC@20 but costs `3.33x` as much under its current OpenRouter route.

## GPT-OSS-120B base result

The unadapted GPT-OSS-120B full-teacher-prompt baseline reaches `0.4789`
Mean-OOD pAUROC@20 and `0.7862` Mean-OOD AUROC. Its exact 8,381.1686-token mean
input and one-token output cost `$0.3270/1K` at the paper's OpenRouter hosted
proxy of `$0.039/M` input and `$0.18/M` output.

GPT-OSS is open-weight and has no first-party hosted price, so this coordinate
is provider- and date-sensitive. The 2026-08-30 OpenRouter headline/lowest
listed `$0.03/$0.17` route projects to `$0.2516/1K`; a `$0.15/$0.60`
higher-priced route projects to `$1.2578/1K`:

| OpenRouter proxy | Input USD / 1K evaluations | Output USD / 1K evaluations | Total USD / 1K evaluations | Frontier at this price? |
| --- | ---: | ---: | ---: | :---: |
| Current headline/lowest listed (`$0.03/$0.17`) | 0.251435 | 0.000170 | **0.251605** | yes |
| Paper-matched (`$0.039/$0.18`) | 0.326866 | 0.000180 | **0.327046** | no |
| Higher-priced route (`$0.15/$0.60`) | 1.257175 | 0.000600 | **1.257775** | no |

At the canonical paper-matched price this is no longer a frontier point. The
Qwen3.5-4B base is `13.1%` cheaper and improves pAUROC@20 from `0.4789` to
`0.6175`, so it strictly dominates this point. The GPT-OSS result remains a
matched model-family baseline rather than an operationally attractive
endpoint.

## GPT-OSS-120B low-reasoning result

Prefilling GPT-OSS's native analysis boundary under `Reasoning: low`, sampling
one frozen trace per row, and then extracting literal binary logits reaches
`0.5578` Mean-OOD pAUROC@20 and `0.7914` Mean-OOD AUROC. This improves the
immediate condition by `0.0789` pAUROC while adding 96.65 conceptual output
tokens per evaluation. The gain is concentrated in AgentDojo, IaC, and
SHADE-Arena; MLE-Sabotage declines.

The exact means are 8,376.1686 conceptual input tokens and 97.6549 output
tokens. The canonical paper-matched `$0.039/$0.18` proxy costs `$0.3442/1K`.
At the 2026-08-31 OpenRouter headline `$0.03/$0.17` route it costs
`$0.2679/1K`; the historical higher `$0.15/$0.60` sensitivity costs
`$1.3150/1K`:

| OpenRouter proxy | Input USD / 1K evaluations | Output USD / 1K evaluations | Total USD / 1K evaluations | Frontier at this price? |
| --- | ---: | ---: | ---: | :---: |
| Current headline/lowest listed (`$0.03/$0.17`) | 0.251285 | 0.016601 | **0.267886** | yes |
| Paper-matched (`$0.039/$0.18`) | 0.326671 | 0.017578 | **0.344248** | no |
| Higher-priced route (`$0.15/$0.60`) | 1.256425 | 0.058593 | **1.315018** | no |

At the canonical rate Qwen3.5-4B base strictly dominates this point: it is
cheaper (`$0.2842/1K`) and stronger (`0.6175`). At the current lowest GPT-OSS
route, low reasoning becomes a frontier point because it is cheaper than Qwen
4B, but it remains well below Qwen3.5-9B's matched-interface `0.6976`. This is
useful evidence that reasoning helps GPT-OSS, not an operational efficiency win.

## Qwen3.5-9B base result

The unadapted Qwen3.5-9B full-teacher-prompt baseline reaches `0.6976`
Mean-OOD pAUROC@20 at `$0.9468/1K`. The exact 9,466.71-token mean contributes
`$0.9467/1K` at OpenRouter's dated `$0.10/M` input price; the one required
output token contributes only `$0.00015/1K` at `$0.15/M`.

This point is not on the observed frontier. The paper's GPT-OSS-120B SFT point
is both cheaper (`$0.511/1K`) and stronger (`0.718`), and the Qwen3-32B SFT+RL
point is also cheaper (`$0.869/1K`) and stronger (`0.703`). The result remains
important as the matched unadapted Qwen base under Gleipnir's full binary
specification; it should not be interpreted as evidence against later trained
Qwen3.5-9B students using the compact deployment prompt.

## Qwen3.5-27B base result

The unadapted Qwen3.5-27B full-teacher-prompt baseline reaches `0.8395`
Mean-OOD pAUROC@20 and `0.9371` Mean-OOD AUROC at `$1.8476/1K`. The exact
9,466.7113-token mean input contributes `$1.8460/1K` at the paper's dated
`$0.195/M` hosted proxy; the one required output token contributes
`$0.00156/1K` at `$1.56/M`. OpenRouter did not list a Qwen3.5-27B route in the
snapshot, so this is explicitly the paper's hosted proxy rather than a current
OpenRouter quote.

This is a new observed frontier point. It improves pAUROC@20 by `0.0595` over
the adjacent lower-cost Nemotron-120B SFT point. It also point-estimate
dominates the paper's Qwen3.5-27B SFT+RL row, improving pAUROC@20 by `0.0085`
while costing `33.4%` less. The next higher-quality point, Kimi-K2.6 SFT,
improves only `0.0015` to `0.841` but costs `5.51x` as much under these dated
hosted-price coordinates.

Within the strictly matched Qwen3.5 base series, the curve rises from `0.6175`
at 4B to `0.6976` at 9B and `0.8395` at 27B. The 9B-to-27B gain (`0.1418`) is
larger than the 4B-to-9B gain (`0.0801`), supporting a steep capacity effect
under this immediate-logit interface. The remaining gap to Kimi K3 is `0.0689`.
That cross-family gap cannot be attributed to parameter count alone because
architecture, pretraining, post-training, and provider inference all change.

## Qwen3.5-9B immediate ordinal result

The matched immediate 0--10 severity interface reaches `0.4858` Mean-OOD
pAUROC@20 and `0.7592` Mean-OOD AUROC at `$0.9628/1K`. It is dominated by the
same model's binary logits, which are cheaper by 1.7% and improve pAUROC@20 by
`0.2118`.

This loss is not principally a mathematical consequence of using 11 bins.
Rounding the saved binary probabilities to the nearest tenth retains `0.6835`
pAUROC@20, only `0.0141` below the continuous baseline. By contrast, the
generated ordinal scores used only seven values, omitted 1, 2, 4, and 6, and
placed 80.6% of examples at 0 or 10. The unadapted model therefore does not use
the requested severity scale as a calibrated drop-in replacement for binary
probabilities.

This comparison isolates immediate output interfaces under matched model and
data conditions. It does not reproduce or reject the paper's trained,
rationale-before-score monitors.

## Qwen3.5-9B visible-reasoning ordinal result

Adding one greedy visible analysis in native non-thinking mode raises the
generated 0--10 interface to `0.6656` Mean-OOD pAUROC@20 and `0.8610`
Mean-OOD AUROC. This recovers 84.9% of the immediate ordinal arm's pAUROC@20
deficit, but remains `0.0320` below continuous binary logits and `0.0179` below
the nearest-tenth binary control.

The result is heterogeneous: reasoning exceeds binary pAUROC@20 on IaC and
Rogue Eval, but loses on the other four sources. It uses all 11 requested
scores rather than the immediate arm's seven, so test-time deliberation changes
the mapping materially rather than merely increasing nominal resolution.

Exact means of 9,628.7113 input tokens and 442.6468 logical output tokens cost
`$1.0293/1K` at the dated Qwen3.5-9B hosted-price snapshot. The reasoned arm is
8.7% more expensive than direct binary logits while ranking worse, so it is
strictly dominated and does not enter the observed frontier. This supports a
narrow partial-recovery conclusion for one greedy non-thinking rationale; it
does not test native thinking, self-consistency, or trained deliberation.

## Qwen3.5-9B visible-reasoning binary-logit result

Adding one greedy visible analysis before extracting the same normalized
literal `0`/`1` logits reaches `0.6923` Mean-OOD pAUROC@20 and `0.8607`
Mean-OOD AUROC. The immediate-logit baseline remains stronger at `0.6976` and
`0.8850`, changes of `-0.0054` and `-0.0243`. Reasoning therefore does not
improve the matched primary endpoint even when the final interface retains
continuous logits rather than generated ordinal bins.

The macro result is not uniform. AgentDojo improves by `0.2980` pAUROC@20,
while the other five sources decline; CUA-SHADE-Arena, MLE-Sabotage, and
SHADE-Arena lose `0.1223`, `0.0816`, and `0.0983`. The score distribution is
also materially different: the reasoned arm has 6,314 unique probabilities,
79.16% hard-decision agreement with direct logits, a `0.2721` mean absolute
score change, and `0.6895` Pearson correlation.

Exact means of 9,473.7113 input tokens and 387.0719 logical output tokens cost
`$1.0054/1K` at the dated Qwen3.5-9B hosted-price snapshot. The arm is 6.2%
more expensive than direct binary logits and slightly weaker, so it is strictly
dominated. This is evidence about one greedy non-thinking rationale under the
full binary specification, not native thinking or trained deliberation.

## Kimi K3 frontier movement

Kimi K3 inserts a new observed frontier point between Kimi-K2.6 SFT and Claude
Sonnet 4.6. It also dominates the paper's GPT-5.4 prompted point: Kimi K3 is
`0.0094` higher in Mean-OOD pAUROC@20 while costing `43.4%` less. Relative to
the adjacent higher-quality Sonnet point, Kimi is `2.34x` cheaper for a `0.0206`
lower pAUROC@20.

Sonnet is a useful same-price reference. Its paper price and the frozen Kimi K3
Fireworks ceiling are both `$3/M` input and `$15/M` output, so the observed cost
difference is entirely token accounting:

| Monitor or counterfactual | Mean input tokens | Mean output tokens | Input USD / 1K evaluations | Output USD / 1K evaluations | Total USD / 1K evaluations |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kimi K3 binary logits | 8,322.55 | 10.0 | 24.9677 | 0.1500 | **25.1177** |
| Kimi K3 with Kimi-K2.6's 780.9-token output | 8,322.55 | 780.9 | 24.9677 | 11.7135 | **36.6812** |
| Kimi K3 with Sonnet's 1,029.3-token output | 8,322.55 | 1,029.3 | 24.9677 | 15.4395 | **40.4072** |
| Claude Sonnet 4.6 prompted | 14,470.3 | 1,029.3 | 43.4109 | 15.4395 | **58.8504** |

Matching Kimi-K2.6's output length would therefore make Kimi K3 `1.46x`, not
`2x`, as expensive as the immediate-logit run. Doubling the current Kimi cost
at the same input length and rates would require about 1,685 billed output
tokens. Of the `$33.73/1K` observed gap between Kimi logits and Sonnet, about
`$15.29/1K` is the avoided output and `$18.44/1K` is the lower measured input
token count. Attribute only the former directly to immediate score extraction;
the latter also reflects prompt representation and tokenizer differences.

The defensible result is therefore interface-level: a full-specification binary
logit monitor reaches approximately the same high-quality region with much less
generation, moving the observed cost--performance frontier. It is not evidence
that Kimi K3 would dominate the same models under their original generated-score
interface.

## Adding future Gleipnir evaluations

Add a row only after all of the following are true:

1. The evaluation covers all 6,395 frozen OOD row IDs exactly once, with failed
   requests recovered or explicitly invalidating the run.
2. The prompt, model/provider revision, request settings, score extraction,
   dataset revision, and artifact hashes are recorded in a finding document.
3. Mean-OOD uses the unweighted six-source raw normalized pAUROC@20 above. Keep
   pooled and Weighted-OOD metrics in the finding, not in this coordinate.
4. Cost uses exact saved or tokenizer-rendered mean token counts and the dated
   uncached list prices. Add a new row rather than revising history when prices,
   providers, prompts, or interfaces change.
5. OOD labels did not select the prompt, checkpoint, threshold, subset, retry
   inclusion rule, or stopping point.
6. Recompute `Frontier` against every retained row using unrounded metrics when
   available. A new point that dominates an old point changes the old row to
   `no`; it does not remove that row.

Use this row shape:

```text
| Gleipnir | <model and training> | <prompt and score interface> |
  <uncached USD/1K> | <six-source Mean-OOD pAUROC@20> | <yes/no> |
```

For trained stochastic arms, enter the predeclared aggregate over three paired
seeds. For deterministic literal-logit API baselines, record the single-run
nature explicitly rather than presenting a bootstrap interval as seed
variability.

## Sources

- [Sinha et al., Appendix D.4 metric definition](https://arxiv.org/html/2605.29601v1#A4.SS4)
- [Sinha et al., Appendix E cost accounting and Table 4](https://arxiv.org/html/2605.29601v1#A5.SS5)
- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3.5-27B model card](https://huggingface.co/Qwen/Qwen3.5-27B)
- [OpenRouter model catalog search for Qwen3.5-4B](https://openrouter.ai/models?q=qwen3.5-4b)
- [OpenRouter model catalog search for Qwen3.5-27B](https://openrouter.ai/models?q=qwen3.5-27b)
- [`tool_trajectory_gpt_oss_ood_benchmark.md`](../findings/tool_trajectory_gpt_oss_ood_benchmark.md)
- [`tool_trajectory_gpt_oss_low_reasoning_ood_benchmark.md`](../findings/tool_trajectory_gpt_oss_low_reasoning_ood_benchmark.md)
- [`tool_trajectory_qwen4b_ood_benchmark.md`](../findings/tool_trajectory_qwen4b_ood_benchmark.md)
- [`tool_trajectory_qwen27b_ood_benchmark.md`](../findings/tool_trajectory_qwen27b_ood_benchmark.md)
- [`tool_trajectory_kimi_ood_benchmark.md`](../findings/tool_trajectory_kimi_ood_benchmark.md)
- [`tool_trajectory_qwen_ood_benchmark.md`](../findings/tool_trajectory_qwen_ood_benchmark.md)
- [`tool_trajectory_qwen_ordinal_ood_benchmark.md`](../findings/tool_trajectory_qwen_ordinal_ood_benchmark.md)
- [`tool_trajectory_inference_economics.md`](tool_trajectory_inference_economics.md)
- [OpenRouter Qwen3.5-9B pricing](https://openrouter.ai/qwen/qwen3.5-9b)
- [OpenRouter GPT-OSS-120B pricing and provider variability](https://openrouter.ai/openai/gpt-oss-120b/performance)
