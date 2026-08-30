# Tool-trajectory OOD cost--performance frontier

Status: living registry, established 2026-08-30.

This file is the canonical comparison table for completed evaluations on the
six-source, 6,395-row strict OOD suite from Sinha et al., *Training Deliberative
Monitors for Black-Box Scheming Detection*. Add every future Gleipnir OOD
evaluation here once coverage and cost accounting pass the gates below. Do not
use canaries, incomplete runs, or tuned-on-OOD variants as frontier points.

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
places. The Gleipnir rows are deterministic temperature-zero evaluations using
the binary-logit interface. They are comparable in dataset, metric, and
uncached cost convention, but are not interface reproductions of the paper's
generated 0--10 monitor scores.

## Reference table

Rows are ordered by cost. `Frontier` is point-estimate nondominance after adding
all Gleipnir rows currently in this file: no other row is both no more expensive
and at least as accurate, with one strict improvement.

| Origin | Monitor | Evaluation interface | Uncached inference cost (USD / 1,000 evaluations) | Mean-OOD pAUROC@20 | Frontier |
| --- | --- | --- | ---: | ---: | :---: |
| **Gleipnir** | **Qwen3.5-4B base** | **Full binary specification, immediate `0\|1` logits** | **0.2842** | **0.6175** | **yes** |
| **Gleipnir** | **GPT-OSS-120B base** | **Full binary specification, immediate `0\|1` logits** | **0.3270** | **0.4789** | **no** |
| Paper | GPT-OSS-20B SFT | Trained, short prompt, generated rationale + 0--10 score | 0.347 | 0.656 | yes |
| Paper | Qwen3.5-4B SFT+RL | Trained, short prompt, generated rationale + 0--10 score | 0.358 | 0.684 | yes |
| Paper | GPT-OSS-120B SFT | Trained, short prompt, generated rationale + 0--10 score | 0.511 | 0.718 | yes |
| Paper | Nemotron-30B SFT | Trained, short prompt, generated rationale + 0--10 score | 0.590 | 0.578 | no |
| Paper | Qwen3-8B SFT+RL | Trained, short prompt, generated rationale + 0--10 score | 0.748 | 0.628 | no |
| Paper | Qwen3-32B SFT+RL | Trained, short prompt, generated rationale + 0--10 score | 0.869 | 0.703 | no |
| **Gleipnir** | **Qwen3.5-9B base** | **Full binary specification, immediate `0\|1` logits** | **0.9468** | **0.6976** | **no** |
| Paper | Nemotron-120B SFT | Trained, short prompt, generated rationale + 0--10 score | 1.251 | 0.780 | yes |
| Paper | Qwen3.5-27B SFT+RL | Trained, short prompt, generated rationale + 0--10 score | 2.775 | 0.831 | yes |
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
[`tool_trajectory_qwen4b_ood_benchmark.md`](../findings/tool_trajectory_qwen4b_ood_benchmark.md),
[`tool_trajectory_qwen_ood_benchmark.md`](../findings/tool_trajectory_qwen_ood_benchmark.md)
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
- [OpenRouter model catalog search for Qwen3.5-4B](https://openrouter.ai/models?q=qwen3.5-4b)
- [`tool_trajectory_gpt_oss_ood_benchmark.md`](../findings/tool_trajectory_gpt_oss_ood_benchmark.md)
- [`tool_trajectory_qwen4b_ood_benchmark.md`](../findings/tool_trajectory_qwen4b_ood_benchmark.md)
- [`tool_trajectory_kimi_ood_benchmark.md`](../findings/tool_trajectory_kimi_ood_benchmark.md)
- [`tool_trajectory_qwen_ood_benchmark.md`](../findings/tool_trajectory_qwen_ood_benchmark.md)
- [`tool_trajectory_inference_economics.md`](tool_trajectory_inference_economics.md)
- [OpenRouter Qwen3.5-9B pricing](https://openrouter.ai/qwen/qwen3.5-9b)
- [OpenRouter GPT-OSS-120B pricing and provider variability](https://openrouter.ai/openai/gpt-oss-120b/performance)
