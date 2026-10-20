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
live in `src/gleipnir/analysis/plotting.py`; see
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
binary-logit boundary. GPT-6 Luna is one Standard API run at temperature 1,
with reasoning disabled and actual terminal binary token logprobs. They are
comparable in dataset, metric, and uncached cost convention, but do not reproduce
the paper's trained interfaces. Jev 1.13 returns a native binary probability
from raw trajectory state and a frozen Noul question; pplx-decider v1.2 is
scored on the same request through its own readout head. Its information and
instruction interface differs from the full teacher chat prompt; its point
compares complete monitor configurations, not an isolated model substitution.

The suite inherits the paper's 28,000-token input filter, which removed 43% of
SHADE-Arena but at most 4% of any other source. Monitor quality falls with
length inside the retained SHADE-Arena rows, so its scores are optimistic for
the full benchmark; see the
[SHADE-Arena length audit](../findings/shade_length_audit.md).

## Reference table

Gleipnir release models (Gleipnir 4B v0.3 and Gleipnir 27B) are priced at
self-hosted break-even (user decisions, 2026-10-20). The cost is the measured
optimized-serving wall time for all 6,395 rows on one 100%-utilized B200,
times an hourly rate, per 1,000 rows. Each model has two rows:

- **On-demand:** `$7.99/h`, our Runpod rate.
- **Committed GPUs:** `$3.44/h`, CoreWeave's published committed-use ceiling
  (60% off its $8.60 on-demand rate). This is the rate an inference provider
  with a long-term contract could approach.

Rate sources are in
[B200 hourly rate sources](tool_trajectory_inference_economics.md#b200-hourly-rate-sources).
Other rows keep their list prices or hosted proxies, which include provider
utilization and margin.

Rows are ordered by cost. `Frontier` is point-estimate nondominance: no other
row is both no more expensive and at least as accurate, with one strict
improvement. Each pricing regime has its own frontier: on-demand rows compete
with list-priced rows, and committed-GPU rows compete with list-priced rows.
Committed-GPU rows report their regime's frontier; all other rows report the
on-demand regime. The plot draws both frontiers.

| Origin | Monitor | Evaluation interface | Uncached inference cost (USD / 1,000 evaluations) | Mean-OOD pAUROC@20 | Frontier |
| --- | --- | --- | ---: | ---: | :---: |
| **Gleipnir** | **Qwen3.5-4B Kimi-soft strict seed average (committed GPUs)** | **Compact student specification, immediate `0\|1` logits; QAD, optimized FP8/FP4/MXFP8 serving; self-hosted committed-use B200 break-even** | **0.0366** | **0.7703** | **yes** |
| **Gleipnir** | **Qwen3.5-4B Kimi-soft strict seed average** | **Compact student specification, immediate `0\|1` logits; QAD, optimized FP8/FP4/MXFP8 serving; self-hosted on-demand B200 break-even** | **0.0850** | **0.7703** | **yes** |
| **Gleipnir** | **Qwen3.8-27B Kimi-soft strict (committed GPUs)** | **Full binary specification, immediate `0\|1` logits; optimized FP8/FP4/MXFP8 serving; self-hosted committed-use B200 break-even** | **0.1677** | **0.9508** | **yes** |
| **Gleipnir** | **pplx-decider v1.2 native readout** | **Local `pplx_decider` readout, raw trajectory state + frozen Jev Noul question; true − false readout logits** | **0.1763** | **0.8956** | **yes** |
| **Gleipnir** | **Qwen3.5-4B base** | **Full binary specification, immediate `0\|1` logits** | **0.2842** | **0.6175** | **no** |
| **Gleipnir** | **GPT-OSS-120B base** | **Full binary specification, immediate `0\|1` logits** | **0.3270** | **0.4789** | **no** |
| **Gleipnir** | **GPT-OSS-120B base** | **Full binary specification, low analysis then `0\|1` logits** | **0.3442** | **0.5578** | **no** |
| Paper | GPT-OSS-20B SFT | Trained, short prompt, generated rationale + 0--10 score | 0.347 | 0.656 | no |
| Paper | Qwen3.5-4B SFT+RL | Trained, short prompt, generated rationale + 0--10 score | 0.358 | 0.684 | no |
| **Gleipnir** | **Jev 1.13 native probabilities** | **Native Decisions API, raw trajectory state + frozen Noul question; native binary probability** | **0.3887** | **0.8495** | **no** |
| **Gleipnir** | **Qwen3.8-27B Kimi-soft strict** | **Full binary specification, immediate `0\|1` logits; optimized FP8/FP4/MXFP8 serving; self-hosted on-demand B200 break-even** | **0.3896** | **0.9508** | **yes** |
| Paper | GPT-OSS-120B SFT | Trained, short prompt, generated rationale + 0--10 score | 0.511 | 0.718 | no |
| Paper | Nemotron-30B SFT | Trained, short prompt, generated rationale + 0--10 score | 0.590 | 0.578 | no |
| Paper | Qwen3-8B SFT+RL | Trained, short prompt, generated rationale + 0--10 score | 0.748 | 0.628 | no |
| **Gleipnir** | **GPT-6 Luna binary logprobs** | **Full binary specification, terminal `0\|1` logprobs, reasoning disabled** | **0.8355** | **0.8365** | **no** |
| **Gleipnir** | **Qwen3.5-9B Kimi-soft mixed** | **Compact student specification, immediate `0\|1` logits** | **0.8659** | **0.8212** | **no** |
| Paper | Qwen3-32B SFT+RL | Trained, short prompt, generated rationale + 0--10 score | 0.869 | 0.703 | no |
| **Gleipnir** | **Qwen3.5-9B base** | **Full binary specification, immediate `0\|1` logits** | **0.9468** | **0.6976** | **no** |
| **Gleipnir** | **Qwen3.5-9B base (ordinal)** | **Full severity specification, immediate 0--10 integer** | **0.9628** | **0.4858** | **no** |
| **Gleipnir** | **Qwen3.5-35B-A3B base (mixed providers)** | **Full binary specification, terminal `0\|1` logits; Darkbloom FP4 then Parasail FP8** | **0.9896** | **0.7580** | **no** |
| **Gleipnir** | **Qwen3.5-9B base (reasoned binary)** | **Full binary specification, visible analysis then `0\|1` logits** | **1.0054** | **0.6923** | **no** |
| **Gleipnir** | **Qwen3.5-9B base (reasoned ordinal)** | **Full severity specification, visible analysis then 0--10 integer** | **1.0293** | **0.6656** | **no** |
| Paper | Nemotron-120B SFT | Trained, short prompt, generated rationale + 0--10 score | 1.251 | 0.780 | no |
| **Gleipnir** | **Qwen3.5-27B base** | **Full binary specification, immediate `0\|1` logits** | **1.8476** | **0.8395** | **no** |
| **Gleipnir** | **Qwen3.8-27B base** | **Full binary specification, immediate `0\|1` logits** | **1.8476** | **0.8901** | **no** |
| Paper | Qwen3.5-27B SFT+RL | Trained, short prompt, generated rationale + 0--10 score | 2.775 | 0.831 | no |
| Paper | GPT-5.4 nano prompted | Full specification, generated rationale + 0--10 score | 3.387 | 0.752 | no |
| Paper | Gemini 3.1 Flash-Lite prompted | Full specification, generated rationale + 0--10 score | 4.157 | 0.815 | no |
| **Gleipnir** | **Kimi K2.6 binary logits** | **Full binary specification, immediate `0\|1` logits** | **7.9110** | **0.9194** | **no** |
| Paper | Kimi-K2.6 SFT | Trained, short prompt, generated rationale + 0--10 score | 10.175 | 0.841 | no |
| Paper | Claude Haiku 4.5 prompted | Full specification, generated rationale + 0--10 score | 19.746 | 0.794 | no |
| **Gleipnir** | **Kimi K3 binary logits** | **Full binary specification, immediate `0\|1` logits** | **25.1177** | **0.9084** | **no** |
| Paper | GPT-5.4 prompted | Full specification, generated rationale + 0--10 score | 44.376 | 0.899 | no |
| Paper | Gemini 2.5 Pro prompted | Full specification, generated rationale + 0--10 score | 45.254 | 0.802 | no |
| Paper | Claude Sonnet 4.6 prompted | Full specification, generated rationale + 0--10 score | 58.850 | 0.929 | no |
| Paper | Gemini 3.1 Pro prompted | Full specification, generated rationale + 0--10 score | 73.218 | 0.940 | no |
| Paper | Claude Opus 4.6 prompted | Full specification, generated rationale + 0--10 score | 94.794 | 0.942 | no |

The 17 paper reference rows reproduce Appendix E, Table 4. Their exact token
counts and prices remain in the paper rather than being duplicated here. The
Gleipnir rows are backed by the completed
[`tool_trajectory_gpt_oss_ood_benchmark.md`](../findings/tool_trajectory_gpt_oss_ood_benchmark.md),
[`tool_trajectory_gpt_oss_low_reasoning_ood_benchmark.md`](../findings/tool_trajectory_gpt_oss_low_reasoning_ood_benchmark.md),
[`tool_trajectory_qwen4b_ood_benchmark.md`](../findings/tool_trajectory_qwen4b_ood_benchmark.md),
[`tool_trajectory_qwen_ood_benchmark.md`](../findings/tool_trajectory_qwen_ood_benchmark.md),
[`tool_trajectory_qwen27b_ood_benchmark.md`](../findings/tool_trajectory_qwen27b_ood_benchmark.md),
[`qwen27b_generations_ood.md`](../findings/qwen27b_generations_ood.md),
[`tool_trajectory_qwen35b_a3b_ood_benchmark.md`](../findings/tool_trajectory_qwen35b_a3b_ood_benchmark.md),
[`tool_trajectory_qwen_ordinal_ood_benchmark.md`](../findings/tool_trajectory_qwen_ordinal_ood_benchmark.md),
[`tool_trajectory_qwen_binary_reasoning_ood_benchmark.md`](../findings/tool_trajectory_qwen_binary_reasoning_ood_benchmark.md),
[`tool_trajectory_qwen_ordinal_reasoning_ood_benchmark.md`](../findings/tool_trajectory_qwen_ordinal_reasoning_ood_benchmark.md)
and
[`tool_trajectory_kimi_ood_benchmark.md`](../findings/tool_trajectory_kimi_ood_benchmark.md) and
[`tool_trajectory_kimi_k26_ood_benchmark.md`](../findings/tool_trajectory_kimi_k26_ood_benchmark.md)
and [`openai_luna_ood_benchmark.md`](../findings/openai_luna_ood_benchmark.md)
and [`tool_trajectory_jev_ood_benchmark.md`](../findings/tool_trajectory_jev_ood_benchmark.md)
and [`pplx_decider_ood_benchmark.md`](../findings/pplx_decider_ood_benchmark.md)
audits, the completed
[`tool_trajectory_distillation_ood_scaling.md`](tool_trajectory_distillation_ood_scaling.md)
record, and the exact accounting in
[`tool_trajectory_inference_economics.md`](tool_trajectory_inference_economics.md).

## Kimi-soft student results

Gleipnir 4B v0.3 starts from the exact weight average of three Qwen3.5-4B
seeds. They were trained on the strict-filtered tool-trajectory data only, the
same recipe as Gleipnir 27B ([seed average](../../experiments/seed_average/README.md)).
Same-model quantization-aware distillation then adapts it to optimized
serving ([QAD](../../experiments/qad_v02/README.md)). Like the 27B, it is
plotted at its optimized-serving score (user decision, 2026-10-20):

| Gleipnir 4B, Mean-OOD | pAUROC@20 | AUROC |
| --- | ---: | ---: |
| **v0.3, optimized serving (plotted)** | **`0.7703`** | `0.9138` |
| v0.3, BF16 serving | `0.8039` | `0.9211` |
| v0.2, BF16 serving | `0.8108` | `0.9199` |
| v0.2, optimized serving without QAD | `0.7579` | `0.9006` |

- **What QAD recovered.** It recovers 98% of the optimized-serving ID loss
  and 23% of the OOD loss.
- **Throughput.** Optimized serving runs at 1.89 times the BF16 throughput
  on the B200 (226K against 120K prompt tokens/s on this workload).
- **Selection.** The seed average was selected by a frozen ID-only rule, and
  OOD was never used for selection.
- **Cost.** It is priced at self-hosted break-even: one B200 at `$7.99/h`,
  fully utilized, scored the full benchmark workload (55,346,879 prompt
  tokens) through optimized serving in 244.9 s, which is `$0.0850/1K`
  (`$0.0098/M` tokens). The earlier hosted proxy, the paper's `$0.03/M` input
  and `$0.15/M` output 4B rate, gave `$0.2598/1K`.
- **Frontier.** It is the lowest-cost frontier point. pplx-decider v1.2 is
  above it at about twice the cost. It dominates the unadapted 4B
  full-prompt baseline: about 30% of the cost, and pAUROC@20 higher by
  `0.1528`.

It replaced the v0.1 point, the full-mixed adapter trained with 13,149
deception rows (`0.7823` Mean-OOD pAUROC@20, `0.9086` AUROC, same cost). The
three individual seeds score `0.7869`, `0.8125` and `0.7768`.

Gleipnir 9B is kept in the table but not plotted (user decision, 2026-10-20):
Gleipnir 4B v0.3 supersedes it, and it is no longer being optimized.

The full-mixed Qwen3.5-9B adapter reaches `0.8212` Mean-OOD pAUROC@20 and
`0.9215` Mean-OOD AUROC. At the frozen dated `$0.10/M` input and `$0.40/M`
output proxy, the same compact prompt costs `$0.8659/1K`. It improves
pAUROC@20 by `0.0388` over the v0.1 mixed 4B at `3.33x` the hosted inference proxy.
GPT-6 Luna dominates it at the dated API prices: `0.8365` at `$0.8355/1K`.
pplx-decider v1.2 (`0.8956` at `$0.1763/1K`) now dominates both mixed 4B
and Jev 1.13; the next frontier point above it is Gleipnir 27B.
The 9B result remains a matched student comparison.

Both adapters were trained for one epoch with rank-128 QLoRA using Kimi K3
soft probabilities only. Teacher targets used the richer teacher prompt;
students trained and evaluated with the compact student prompt. Every row in
the 6,395-example OOD suite was scored, and FP32-master versus rebased-serving
parity passed before full evaluation. These rows report only inference cost;
training cost is excluded under the registry contract.

The 9B mixed adapter does not improve source-balanced OOD performance over the
8,688-row matched-only adapter: Mean-OOD pAUROC changes from `0.8227` to
`0.8212`, and Mean-OOD AUROC is unchanged at `0.9215`. The added deception
data therefore helps neither frontier coordinate nor regular macro ranking at
fixed backbone, despite improving pooled AUROC. Full scaling and source-level
results are in
[`tool_trajectory_distillation_ood_scaling.md`](tool_trajectory_distillation_ood_scaling.md).

## Qwen3.5-4B base result

The unadapted Qwen3.5-4B full-teacher-prompt baseline reaches `0.6175`
Mean-OOD pAUROC@20 and `0.8549` Mean-OOD AUROC. Its exact 9,466.7113-token
mean input and one-token output cost `$0.2842/1K` at the paper's hosted proxy
of `$0.03/M` input and `$0.15/M` output. OpenRouter did not list a Qwen3.5-4B
route on 2026-08-30, so this coordinate is explicitly an Artificial Analysis
proxy carried forward from the paper, not a current OpenRouter quote.

The trained mixed 4B student now strictly dominates this point: it is cheaper
because of its compact prompt and improves pAUROC@20 by `0.1649`. The base
row remains the matched unadapted reference but is no longer on the observed
frontier.

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
| Current headline/lowest listed (`$0.03/$0.17`) | 0.251285 | 0.016601 | **0.267886** | no |
| Paper-matched (`$0.039/$0.18`) | 0.326671 | 0.017578 | **0.344248** | no |
| Higher-priced route (`$0.15/$0.60`) | 1.256425 | 0.058593 | **1.315018** | no |

At every listed rate the Gleipnir 4B student strictly dominates this
point: it costs `$0.0850/1K` and reaches `0.7703` (optimized serving). The GPT-OSS result remains
useful evidence that reasoning helps that model family, not an operational
efficiency win.

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

## Qwen3.5-35B-A3B mixed-provider result

The unadapted sparse model reaches `0.7580` Mean-OOD pAUROC@20 and `0.9050`
Mean-OOD AUROC using the full teacher prompt and terminal binary logprobs.
Darkbloom FP4 served 4,545 rows before a throughput-driven handoff to Parasail
FP8 for the remaining 1,850. The uncached list-price cost is `$0.9896/1K`
using the actual token totals and each provider's frozen rate. This point is
dominated by the lower-cost Gleipnir 4B student (`0.7703` at `$0.0850/1K`), and
falls below the dense Qwen3.5-27B base (`0.8395` at `$1.8476/1K`).

The handoff follows source order: Parasail scored all 850 SHADE-Arena rows and
1,000 Rogue Eval rows, while Darkbloom scored the other four complete sources
and 499 Rogue Eval rows. Provider, FP4/FP8 quantization, and source are thus
confounded; the result cannot establish an isolated sparse-versus-dense model
effect or a single-provider source curve. The paired 100-row non-OOD canary
shows Pearson score correlation `0.947`, Spearman `0.965`, and 90% fixed-0.5
decision agreement. See the [full artifact audit](../findings/tool_trajectory_qwen35b_a3b_ood_benchmark.md).

## Qwen3.5-27B base result

The unadapted Qwen3.5-27B full-teacher-prompt baseline reaches `0.8395`
Mean-OOD pAUROC@20 and `0.9371` Mean-OOD AUROC at `$1.8476/1K`. The exact
9,466.7113-token mean input contributes `$1.8460/1K` at the paper's dated
`$0.195/M` hosted proxy; the one required output token contributes
`$0.00156/1K` at `$1.56/M`. OpenRouter did not list a Qwen3.5-27B route in the
snapshot, so this is explicitly the paper's hosted proxy rather than a current
OpenRouter quote.

Jev 1.13 now dominates this point at `0.8495` and `$0.3887/1K`. Qwen27B
remains stronger than GPT-6 Luna by `0.0029` pAUROC and mixed Qwen3.5-9B by
`0.0183`. It also point-estimate
dominates the paper's Qwen3.5-27B SFT+RL row, improving pAUROC@20 by `0.0085`
while costing `33.4%` less. The next higher-quality frontier point, Kimi K2.6
binary logits, improves by `0.0799` to `0.9194` at `$7.9110/1K` under these
dated hosted-price coordinates.

Within the strictly matched Qwen3.5 base series, the curve rises from `0.6175`
at 4B to `0.6976` at 9B and `0.8395` at 27B. The 9B-to-27B gain (`0.1418`) is
larger than the 4B-to-9B gain (`0.0801`), supporting a steep capacity effect
under this immediate-logit interface. The remaining gap to Kimi K3 is `0.0689`.
That cross-family gap cannot be attributed to parameter count alone because
architecture, pretraining, post-training, and provider inference all change.

## Qwen3.8-27B base result

The unadapted Qwen3.8-27B (revision `1d4bf0f2…`, BF16, vLLM 0.31 on B200) reaches
`0.890067` Mean-OOD pAUROC@20 and `0.9535` Mean-OOD AUROC under the frozen
Qwen3.5-27B full-teacher-prompt contract
([finding](../findings/qwen27b_generations_ood.md)). It renders every prompt
byte-identically to Qwen3.5-27B, so the mean input is the same
9,466.7113 tokens plus one output token.

Qwen3.8-27B has the same architecture and parameter count as Qwen3.5-27B
(`Qwen3_5ForConditionalGeneration`, 27.78B), so its cost uses the same paper
hosted proxy (`$0.195/M` input, `$1.56/M` output): `$1.8476/1K`. The two 27B
points therefore differ only in quality. For reference, OpenRouter listed
`qwen/qwen3.8-27b` at `$0.24/M` input and `$2.20/M` output on 2026-10-17, which
would give `$2.2742/1K`.

The point is new frontier: Jev 1.13 is the best cheaper row (`0.8495`), and the
next higher-quality frontier point is Kimi K2.6 (`0.9194` at `$7.9110/1K`). It
dominates none of the previous frontier points; at equal cost it is `0.0506`
above Qwen3.5-27B. It is `0.0183` below Kimi K3 at about 1/14 of Kimi K3's
cost.

## Qwen3.8-27B Kimi-soft result

Qwen3.8-27B trained for one epoch on the strict-filtered Kimi K3 soft targets
(seed 0, `qwen38_27b_b200_default`; campaign `qwen38-27b-s0-01`) is entered
under the full teacher prompt, the interface of the unadapted 27B and Kimi
rows. It is plotted at its optimized-serving score. Gleipnir models are
entered as they will be served (user decision, 2026-10-20). Cost stays at the
dated proxy until project serving prices are set.

- **Optimized serving.** `QuantFullWorker`, with FP8 attention projections,
  FP4 MLP and GDN, MXFP8 attention and CUDA graphs to 2,048 tokens, and no
  QAD. It reaches `0.9508` Mean-OOD pAUROC@20 and `0.9841` Mean-OOD AUROC
  ([optimized gap](../../experiments/optimized_gap/README.md)).
- **BF16 reference.** Stock vLLM 0.31 BF16 with an FP32 LM head reaches
  `0.9352` and `0.9790`
  ([experiment](../../experiments/gleipnir27b_vocab_distillation/README.md)).
- **Where the difference is.** The `+0.0156` difference is almost entirely
  SHADE-Arena (`0.8316` against `0.7298`); AgentDojo moves `−0.0073` and the
  other four sources at most `0.0008`. SHADE-Arena is one-task-dominated, so
  read this as a serving perturbation that happens to help, not as a quality
  gain.
- **QAD.** Under the compact student prompt the same stack moves the 27B by
  `−0.0025`, so it needs no QAD.
The prompt was chosen on CoT-removed ID (0.9420 vs. 0.9216 under the compact student
prompt it was trained on), not on OOD. The workload renders byte-identically
to the unadapted 27B row (60,539,619 input tokens, mean 9,466.7113, plus one
output token). It is priced at self-hosted break-even: one B200 at `$7.99/h`,
fully utilized, scored this workload through optimized serving in 1,122.6 s,
which is `$0.3896/1K` (`$0.0412/M` tokens). The shared 27B hosted proxy
(`$0.195/M` input, `$1.56/M` output) gave `$1.8476/1K`. Training cost is
excluded under the registry contract.

This row replaces the campaign's compact-student-prompt entry (`0.9087`
Mean-OOD pAUROC@20, `0.9679` AUROC, `$1.6892/1K`;
[finding](../findings/qwen38_27b_campaign.md)). Almost all of the gain is
SHADE-Arena (`0.7298` vs. `0.5875`); the other five sources move by at most
`0.011`.

It dominates the unadapted Qwen3.8-27B, which is still at the hosted proxy
(`0.8901` at `$1.8476/1K`): `0.0607` higher (`0.0451` for the BF16 score).
It also dominates Kimi K2.6 (`0.9194` at `$7.9110/1K`) and Claude Sonnet 4.6
(`0.929` at `$58.850/1K`), which leave the frontier, and Kimi K3 (`0.9084`).
At its optimized-serving score it is the highest-scoring row and also
dominates Gemini 3.1 Pro (`0.940` at `$73.218/1K`) and Claude Opus 4.6
(`0.942` at `$94.794/1K`), which leave the frontier. They cost about 190–240
times as much. At the BF16 score (`0.9352`) both would remain above it. One seed; the 4B students' OOD seed SD is 1.8 points, and
SHADE-Arena is dominated by one task
([length audit](../findings/shade_length_audit.md)).

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

## GPT-6 Luna Standard API frontier movement

The complete, audited single-run benchmark reaches `0.8365439238` Mean-OOD
pAUROC@20 with the full teacher prompt, no reasoning, and terminal binary
logprobs. Standard short-context prices checked on 2026-09-30 are `$0.10/M`
input and `$0.50/M` output. The 6,395 accepted responses report exactly
53,206,252 input and 44,766 billed output tokens, averaging 8,319.976857 and
7.000156 tokens. The primary uncached coordinate is **`$0.835497764/1K`**:

```text
((53,206,252 * 0.10 + 44,766 * 0.50) / 1,000,000) * 1,000 / 6,395
```

This follows the registry's successful-evaluation convention. It excludes
the three failed HTTP-200 responses, all HTTP-429 attempts, non-OOD canaries,
cache reads and cache-write premiums. Those operational costs remain in the
[benchmark finding](../findings/openai_luna_ood_benchmark.md). Including OOD
recovery usage instead would give `$0.836167/1K`; that is not the plot coordinate.

Before Jev's retained result was added, Luna became the next frontier point
after Gleipnir 4B. It dominates Gleipnir
9B's dated hosted proxy: it costs about 3.5% less while improving pAUROC by
`0.0153`. Qwen3.5-27B has `0.0029` higher pAUROC at about 2.21 times the cost.
Jev 1.13 now dominates both points at `0.8495` and `$0.3887/1K`.
These are point-estimate comparisons, without a
paired significance claim, and hosted list prices do not establish self-hosted
cost or a deployment preference.

Luna's SHADE-Arena FPR is 91.67% at the frozen 0.5 threshold. Ranking and cost
alone therefore do not establish a calibrated deployment monitor. Top-20
logprobs also failed to include both literal scores once; that response and
two output-cap failures recovered on first identical repeats. All raw failures
and the recorded bounded recovery policy are retained in the audit.

GPT-5.6 Luna is excluded from this registry and plot at the user's request;
its completed benchmark remains in its separate finding.

## Jev 1.13 native-probability frontier movement

At the user's request on 2026-09-30, add the retained 2026-09-19 evaluation
of `typesafe/jev-1.13`, served as `typesafe/jev-1.13-20260917` by TypeSafe.
All 6,395 frozen IDs are present exactly once; saved states match the original
full trajectories. An independent artifact audit reproduces all source metrics,
the request/config/question hashes, and the provider-reported cost.

Jev reaches `0.8495130129` Mean-OOD pAUROC@20 and `0.9268991833` Mean-OOD
AUROC. At the recorded `$0.042/M` input and `$0/M` output rates, its
59,178,752 input tokens cost `$2.485507584` over the suite, or
**`$0.388664204/1K`**. Reported output usage is 147,085 tokens; the output
charge is zero. The native probability comes from raw trajectory state plus
the frozen Noul question rather than full teacher chat-prompt logprobs.

Jev inserted a frontier point between Gleipnir 4B and Kimi K2.6, dominating
GPT-6 Luna, Gleipnir 9B, and Qwen27B on these dated point estimates.
pplx-decider v1.2 later dominated it on the same requests. It retains
95 unique scores, so ties are much more frequent than for token-logprob
monitors. SHADE-Arena remains weak at `0.4415` pAUROC. Two full responses
report input usage slightly above the configured 32,000-token window; saved
records do not establish whether the provider internally truncated them.
The [finding](../findings/tool_trajectory_jev_ood_benchmark.md) preserves this
context discrepancy and the different native interface. Do not infer a
prompt-matched model advantage or calibrated deployment performance.

## pplx-decider v1.2 native-readout frontier movement

At the user's request on 2026-10-19, Perplexity's open `pplx-decider-v1.2`
(Qwen3.8-27B backbone, revision `cce0f3f8…`) was scored locally on all 6,395
frozen IDs. It ran in its own `pplx_decider` runtime, with the exact Jev
states and Noul question. The readout logit margin reaches `0.8955830038`
Mean-OOD pAUROC@20 and `0.9576814579` Mean-OOD AUROC.

Cost: 56,359,499 local joint-sequence input tokens at the OpenRouter list
price of the `-20261016` endpoint (`$0.02/M` input, no output charge, no
caching) give **`$0.176261/1K`**. The hosted endpoint's serving and billing
are not observed.

It is an on-demand frontier point between the two self-hosted Gleipnir models: above
Gleipnir 4B v0.3 (`0.7703` at `$0.0850/1K`) at about twice its cost, and below
Gleipnir 27B (`0.9508` at `$0.3896/1K`). It dominates Jev 1.13 (`0.8495` at
`$0.3887/1K`) and beats it on all six sources on identical requests. Its cost
is a hosted list price, while the Gleipnir rows are self-hosted break-even
at full utilization. SHADE-Arena stays weakest at
`0.6054`.

**Self-hosting it costs far more than its list price.**
- **Measured.** The published fast path (packed batches, `torch.compile`,
  CUDA graphs) scored the full workload on our B200 at 13,515 prompt
  tokens/s (56,359,499 tokens in 4,170 s).
- **Break-even.** At the same `$7.99/h`, 100%-utilization rule as the
  Gleipnir rows, that is `$0.164/M` tokens, or `$1.447/1K` evaluations:
  8.2 times the `$0.02/M` list price.
- **Flash.** pplx-decider-flash v1.2 behaves the same way: 30,655 tokens/s,
  `$0.072/M`, 7.2 times its `$0.01/M` price.
- **What the list price implies.** Matching `$0.02/M` at `$7.99/h` would
  need about 111K tokens/s on one B200, or about 47.8K at the `$3.44/h`
  provider estimate, where the published fast path costs `$0.0707/M`
  (`$0.6231/1K`). For comparison, Gleipnir 27B (same
  Qwen3.8-27B backbone, causal attention) reaches 53.9K tokens/s on our
  optimized stack, at which the Decider would break even at about
  `$0.041/M` at `$7.99/h` or `$0.0177/M` at `$3.44/h`. The Decider's bidirectional attention and custom readout also
  keep it out of standard causal serving stacks.
- **Conclusion.** The list price is feasible with discounted GPUs and a
  serving stack about 3.5 times faster than the published one. Otherwise it
  is priced below cost.
- **Committed GPUs.** Against Gleipnir rows at committed-GPU rates, the
  Decider leaves the frontier: Gleipnir 27B (`0.9508` at `$0.1677/1K`) is
  both cheaper and better.
- **Plotting.** The plot keeps the list price, which is what users pay.
  Plotted at self-hosted break-even, it would sit well behind Gleipnir 27B.
- **Caveat.** Only the documented runtime configuration was measured, with
  no batch-size tuning or quantization.
- **Receipts.** `results/pplx_decider_ood/{decider,flash}/full_fast.receipt.json`.

The 4B sibling `pplx-decider-flash-v1.2` scores `0.5350` at `$0.0881/1K`.
At the user's request it is not added, because it falls below base
Qwen3.5-4B (`0.6175`). See the
[finding](../findings/pplx_decider_ood_benchmark.md) for identities,
fast-path qualifications and the Flash teacher-specification arm.

## Kimi K2.6 Inceptron frontier movement

The full-specification Kimi K2.6 binary-logit monitor reaches `0.9194`
Mean-OOD pAUROC@20 at a paper-priced `$7.9110/1K`. It becomes the
highest-quality observed open-weight frontier point. It displaces both the
paper's Kimi-K2.6 SFT point and Gleipnir's Kimi K3 point. Relative to the
paper's K2.6 SFT monitor, it gains
`0.0784` pAUROC while costing 22.3% less. Relative to Kimi K3, it gains
`0.0110` while costing 68.5% less.

The next higher-quality point is Claude Sonnet 4.6 at `0.929`; K2.6 is
`0.0096` lower but `7.44x` cheaper. This is a much larger frontier movement
than the small macro-score difference alone suggests.

The K2.6-over-K3 macro gain is source-specific rather than uniform. K2.6 loses
on AgentDojo, CUA-SHADE, IaC, and Rogue Eval, ties on MLE-Sabotage, and improves
SHADE-Arena pAUROC by `0.1502`. K3 remains stronger on pooled pAUROC and fixed-
threshold diagnostics. Retain the source-balanced frontier claim because that
is the frozen paper estimand, but do not describe K2.6 as uniformly stronger.

Exact usage is 53,145,980 prompt tokens and 25,580 completion tokens. At the
paper's Kimi-K2.6 `$0.95/M` input and `$4.00/M` output prices, the primary
full-suite coordinate is `$50.5910`, or `$7.9110/1K`. At Inceptron's dated
route prices, the same usage is `$28.2541` uncached; OpenRouter reported
`$22.9309` after a 26.61% cache-read fraction. Both operational figures are
excluded from the paper-comparable coordinate. See the dedicated
[`run audit`](../findings/tool_trajectory_kimi_k26_ood_benchmark.md).

## Kimi K3 historical frontier movement

Before the K2.6 Inceptron result, Kimi K3 inserted a new observed frontier point
between Kimi-K2.6 SFT and Claude Sonnet 4.6. It also dominated the paper's
GPT-5.4 prompted point: Kimi K3 is
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
generation. Kimi K3 is now dominated by the cheaper, stronger K2.6 Inceptron
point, but the historical comparison still isolates generation economics at a
shared K3/Sonnet price. It is not evidence
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

The [strict-filtering transfer diagnostic](../findings/teacher_disagreement_ood.md)
completed all 6,395 rows at Mean-OOD pAUROC@20 `0.786936`. It remains outside
the frontier table because it is a single trained seed, compared against cached
historical models, rather than the required three-paired-seed aggregate.

## Sources

- [Gleipnir 4B release](https://huggingface.co/Jazhyc/Gleipnir-4B)
- [Gleipnir 9B release](https://huggingface.co/Jazhyc/Gleipnir-9B)
- [Sinha et al., Appendix D.4 metric definition](https://arxiv.org/html/2605.29601v1#A4.SS4)
- [Sinha et al., Appendix E cost accounting and Table 4](https://arxiv.org/html/2605.29601v1#A5.SS5)
- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3.5-27B model card](https://huggingface.co/Qwen/Qwen3.5-27B)
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- [OpenRouter Qwen3.8-27B listing](https://openrouter.ai/qwen/qwen3.8-27b)
- [OpenRouter model catalog search for Qwen3.5-4B](https://openrouter.ai/models?q=qwen3.5-4b)
- [OpenRouter model catalog search for Qwen3.5-27B](https://openrouter.ai/models?q=qwen3.5-27b)
- [`tool_trajectory_gpt_oss_ood_benchmark.md`](../findings/tool_trajectory_gpt_oss_ood_benchmark.md)
- [`tool_trajectory_gpt_oss_low_reasoning_ood_benchmark.md`](../findings/tool_trajectory_gpt_oss_low_reasoning_ood_benchmark.md)
- [`tool_trajectory_qwen4b_ood_benchmark.md`](../findings/tool_trajectory_qwen4b_ood_benchmark.md)
- [`tool_trajectory_qwen27b_ood_benchmark.md`](../findings/tool_trajectory_qwen27b_ood_benchmark.md)
- [`tool_trajectory_kimi_ood_benchmark.md`](../findings/tool_trajectory_kimi_ood_benchmark.md)
- [`openai_luna_ood_benchmark.md`](../findings/openai_luna_ood_benchmark.md)
- [`tool_trajectory_jev_ood_benchmark.md`](../findings/tool_trajectory_jev_ood_benchmark.md)
- [`pplx_decider_ood_benchmark.md`](../findings/pplx_decider_ood_benchmark.md)
- [pplx-decider-v1.2 model card](https://huggingface.co/perplexity-ai/pplx-decider-v1.2)
- [OpenRouter pplx-decider-v1.2 listing](https://openrouter.ai/perplexity/pplx-decider-v1.2)
- [`tool_trajectory_qwen_ood_benchmark.md`](../findings/tool_trajectory_qwen_ood_benchmark.md)
- [`tool_trajectory_qwen_ordinal_ood_benchmark.md`](../findings/tool_trajectory_qwen_ordinal_ood_benchmark.md)
- [`tool_trajectory_inference_economics.md`](tool_trajectory_inference_economics.md)
- [OpenRouter Qwen3.5-9B pricing](https://openrouter.ai/qwen/qwen3.5-9b)
- [OpenRouter GPT-OSS-120B pricing and provider variability](https://openrouter.ai/openai/gpt-oss-120b/performance)
