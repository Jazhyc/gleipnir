# Qwen 27B generations on the strict OOD suite

This experiment chooses which dense Qwen 27B checkpoint to use as the
starting point for the planned 27B student. All three checkpoints share the
`Qwen3_5ForConditionalGeneration` architecture and the 27.78B parameter count.

| Model | Revision |
|---|---|
| `Qwen/Qwen3.5-27B` | `fc05daec18b0a78c049392ed2e771dde82bdf654` |
| `Qwen/Qwen3.6-27B` | `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9` |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

Only Qwen3.5-27B has an OOD result: 0.8395 Mean-OOD pAUROC@20, from 2026-08-31
([finding](../../docs/findings/tool_trajectory_qwen27b_ood_benchmark.md)). The
earlier Qwen3.8 attempt was the FP8 checkpoint on vLLM 0.24 and H100. It
failed its batch/singleton consistency gate, so no quality result exists for
it ([ID screen](../qwen122b_id/README.md)). This run uses BF16 throughout.

## Interface audit (`audit.py`, CPU only)

The checkpoints ship different `chat_template.jinja` files, and Qwen3.8 also
ships a different `tokenizer.json`. Despite that, all three render all 6,395
frozen prompts byte-identically (SHA-256 `a71a9ee6…`). Token counts are also
identical: 60,539,619 in total, maximum 30,382, matching the frozen audit. The
decision tokens are the same in all three: `0`/`1` = [15, 16].

## Design (frozen before scoring)

**Contract.** Each config copies `qwen27b_ood_benchmark.json` and changes only
the model, revision, campaign id and hypothesis text. The following are
byte-for-byte the frozen 3.5 contract:
- the full teacher prompt;
- the non-thinking boundary and `Prediction:`;
- literal `0`/`1` logprob scoring;
- every engine setting: 32,768 maximum length, 16 sequences,
  32,768-token batches, prefix caching, seed 20260830.

**Stack.** The runs use the EU-RO-1 B200 Pod `q7io6ui6gwbqo6` with vLLM 0.31
and BF16 weights on the container disk. They run in order: 3.5, then 3.6, then
3.8.

**Drift control.** The 3.5 rerun is scored against its 2026-08-31 predictions,
which used vLLM 0.24 on an RTX PRO 6000. The comparison reports score MAE,
correlation and Mean-OOD pAUROC@20 difference. Generations are compared only
within this run.

**Reporting.** Results are descriptive; the user selects the starting
checkpoint. The report covers:
- Mean-OOD pAUROC@20, the paper aggregate: the unweighted mean of six
  per-source values;
- per-source pAUROC@20 and AUROC;
- macro Brier, balanced accuracy and score ties;
- per-run scoring time.

No prompt, threshold or setting is tuned on these labels.

**Stop** on any frozen stop condition: input, prompt, boundary or revision
drift; truncation; missing decision logprobs; an incomplete or non-finite
12-row canary; or missing or duplicate rows.

```bash
PYTHONPATH=src:. python -m experiments.qwen27b_generations_ood.audit
HF_HOME=/root/hf-27b HF_HUB_OFFLINE=1 PYTHONPATH=src:. <vllm-0.31-python> \
  -m experiments.tool_trajectory_monitoring.benchmark_qwen_ood \
  --config experiments/qwen27b_generations_ood/qwen36_27b.json \
  --output results/qwen27b_generations_ood/qwen36_27b
```
