# GPT-OSS-120B low-reasoning OOD benchmark

Date: 2026-08-31. Status: completed on all 6,395 strict OOD rows.

## Frozen design

This condition uses the same pinned `openai/gpt-oss-120b` revision, complete
teacher prompt, and six-source OOD artifact as the immediate-logit baseline.
The Harmony header requests `Reasoning: low`; the evaluator prefills the native
analysis boundary, samples one trace at temperature 1 and top-p 1 with seed
`20260830`, then appends the canonical final `Prediction:` boundary. The next
token is constrained to literal `0` or `1`, both logprobs are retained, and
their normalized two-class probability is the score.

The prompt-template SHA-256 is
`2418cc55801deead8983d3bde6e35b59603f0080711ce8e12e1d7299487ac128`,
the input SHA-256 is
`fa4b0d917e51eb91a52efa93cddf0eb154b88993269c97cfdb161c42680136a9`,
and the final config SHA-256 is
`474b2c0be5f87cbaf42a575113c7d5ce4605775fc6a726ce6987ab336f0f3c28`.
GPT-OSS's otherwise dynamic Harmony date is frozen to `2026-08-30`. OOD labels
did not select any prompt, trace, setting, threshold, subset, retry inclusion
rule, or stopping point.

## Result

| Population | N | AUROC | pAUROC@20 |
| --- | ---: | ---: | ---: |
| **Mean-OOD** | **6 sources** | **0.7914** | **0.5578** |
| Pooled examples | 6,395 | 0.7984 | 0.5303 |
| AgentDojo | 1,500 | 0.9511 | 0.8271 |
| CUA-SHADE-Arena | 955 | 0.8031 | 0.6021 |
| IaC | 1,443 | 0.7380 | 0.4577 |
| MLE-Sabotage | 148 | 0.9181 | 0.8219 |
| Rogue Eval | 1,499 | 0.6152 | 0.2123 |
| SHADE-Arena | 850 | 0.7231 | 0.4259 |

Relative to immediate logits, Mean-OOD pAUROC@20 improves by `0.0789`; pooled
AUROC improves by `0.0704`. The largest source pAUROC gains are IaC (`+0.2134`)
and AgentDojo (`+0.1872`), while MLE-Sabotage declines by `0.0724`. There are
1,449 threshold prediction flips, mean absolute score movement is `0.2151`, and
paired score correlation is only `0.4647`. Low reasoning materially changes
the monitor but does not close the gap to Qwen3.5-9B's `0.6976` Mean-OOD
pAUROC@20.

At threshold 0.5, source-macro balanced accuracy is `0.7445`, Brier score is
`0.23638`, recall is `0.5268`, and FPR is `0.0377`. The 6,395 rows have only
554 unique scores, so ties remain material.

## Exact cost

The one-pass hosted equivalent has 53,565,598 input tokens and 624,503 output
tokens: means of 8,376.1686 and 97.6549 per evaluation. Output includes the
prefilled analysis boundary as conceptual provider output, sampled analysis,
Harmony transitions, and one decision token. It does not double-charge the
local second prefill used to extract logits.

At the paper-matched OpenRouter proxy of `$0.039/M` input and `$0.18/M` output:

```text
input  = 8,376.1686 * $0.039 / 1000 = $0.326671 per 1K evaluations
output =    97.6549 * $0.18  / 1000 = $0.017578 per 1K evaluations
total                                      $0.344248 per 1K evaluations
```

The canonical row records **`$0.3442/1K`**. The 2026-08-31 OpenRouter headline
`$0.03/$0.17` route gives `$0.2679/1K`; a historical `$0.15/$0.60` sensitivity
gives `$1.3150/1K`. These are hosted token-price proxies, not local Slurm costs.

## Failure and artifact audit

The final v3 job `30979917` completed in 1 hour 20 minutes 22 seconds on one
NVIDIA RTX PRO 6000 Blackwell Server Edition with vLLM 0.24.0 and PyTorch
2.11.0+cu130. All IDs occur exactly once; there are zero empty analyses,
format failures, truncations, missing logprobs, invalid decisions, or
non-finite scores. Mean sampled analysis content is 85.65 tokens and the
maximum is 1,270.

Two superseded launches failed closed and are excluded. V1 let the model choose
its first channel; one of 4,172 rows skipped analysis and emitted an unrelated
free-form final answer. V2 prefixed the analysis channel and generated all
6,395 valid traces, but crossed midnight: the upstream template's direct
`strftime_now` call placed `2026-08-30` in 4,108 prompts and `2026-08-31` in
2,287. V3 froze the former date, retained only the 4,108 exact matches, and
regenerated the other 2,287. Failed artifacts remain in separate ignored
directories and do not enter the result.

Final ignored artifact hashes are:

- `generations.jsonl`: `8a92b56646b7616857afb09278cf4b4e347634567e0220a04fa593ad138d239f`;
- `predictions.jsonl`: `e7e1d544624adf65dd9077541744bda5891a10976a5fa1e453a1f28e248ed077`;
- `canary_result.json`: `8885ecfbe66d6cf576dcd10bb3e59d5e45d52b664366dc37f82784578b72d493`;
- `result.json`: `d82f345c4f0829b101622a08c70b32c9447fb423caf49d93da3d9d95833b81e3`.

## Interpretation

Lowest-effort reasoning helps the unadapted GPT-OSS monitor, especially at low
FPR on AgentDojo and IaC, but its improvement is heterogeneous and insufficient
to explain the Qwen3.5-9B gap. At the canonical paper-matched price, Qwen3.5-4B
base strictly dominates it: `$0.2842/1K`, `0.6175` versus `$0.3442/1K`,
`0.5578`. At today's lowest GPT-OSS route the point is technically on the
frontier because it costs only `$0.2679/1K`. This is a provider-price
technicality, not evidence that low-reasoning GPT-OSS is the preferred monitor.
