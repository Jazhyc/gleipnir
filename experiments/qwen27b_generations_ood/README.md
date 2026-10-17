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

## Results (2026-10-16/17, B200 `GPU-3d8d6bde…`, vLLM 0.31.0)

Every run scored all 6,395 rows, passed its canary, and hit no stop condition.
Each scoring invocation took about 2,190 s (36.5 min), and engine start-up
took about 64 s.

| strict OOD | Qwen3.5-27B | Qwen3.6-27B | **Qwen3.8-27B** |
|---|---:|---:|---:|
| **Mean-OOD pAUROC@20** | 0.8388 | 0.8726 | **0.8901** |
| macro AUROC | 0.9369 | **0.9538** | 0.9535 |
| macro Brier | 0.1067 | 0.1050 | **0.0823** |
| macro balanced accuracy | 0.8659 | 0.8734 | **0.8894** |
| tied rows | 2,360 | 2,107 | **1,837** |

| pAUROC@20 by source | 3.5 | 3.6 | 3.8 |
|---|---:|---:|---:|
| AgentDojo | 0.9134 | 0.9446 | **0.9657** |
| CUA-SHADE-Arena | 0.8151 | 0.8513 | **0.8648** |
| IaC | 0.8860 | 0.8754 | **0.9678** |
| MLE-Sabotage | 0.9908 | 0.9990 | **1.0000** |
| Rogue Eval | 0.9340 | 0.9694 | **0.9862** |
| SHADE-Arena | 0.4935 | **0.5957** | 0.5560 |

**Drift control.** The 3.5 rerun matches its 2026-08-31 result:
- Mean-OOD pAUROC@20 is 0.8388 vs. 0.8395, and every source is within 0.003;
- row scores correlate at 0.9997, with MAE 0.0039;
- 30 rows flip at the 0.5 threshold.

So the stack change from vLLM 0.24 on an RTX PRO 6000 to vLLM 0.31 on a B200
does not affect the comparison.

**Reading.**
- **Qwen3.8 ranks first.** It is best on five of six sources and on
  calibration. It is 5.1 points above 3.5 and 1.8 above 3.6. The gap to the
  Kimi K3 teacher (0.9084) narrows from 6.9 points (3.5) to 1.8.
- **Qwen3.6 leads only on SHADE-Arena:** 0.596 vs. 0.556. SHADE-Arena remains
  every model's weakest source.
- **Limits of the evidence.**
  - These are single unadapted checkpoints at a fixed prompt interface.
  - ID is scored below; the training-row check is in-distribution to Kimi's
    targets and is not a held-out result.
  - The BF16 run did not repeat the earlier FP8 batch/singleton consistency
    gate. The earlier FP8 Qwen3.8 consistency failure has not been checked for
    BF16.

## Qwen3.8-27B on the ID and training rows (2026-10-17)

`id_train_configs.py` freezes two configs that change only the input rows of
the Qwen3.8 contract above:
- ID: 3,012 rows;
- training: all 8,688 rows.

Both runs scored every row and passed the canary (ID 1,519 s; training
3,714 s). `compare_id_train.py` joins the training predictions with the Kimi
K3 soft targets by row id and checks `rendered_prompt_sha256`
(414 prompts are shared by more than one row) and writes
`results/qwen27b_generations_ood/id_train_comparison.json`. Metrics are
source-macro; the threshold is 0.5.

**ID (3,012 rows).** Unadapted Qwen3.8-27B against the trained 4B BF16/SDPA
students:

| | Qwen3.8-27B | 4B seed 0 | 4B seed 1 | 4B seed 2 |
|---|---:|---:|---:|---:|
| macro pAUROC@20 | 0.8869 | 0.8959 | 0.8978 | 0.9142 |
| macro AUROC | 0.9730 | 0.9638 | 0.9627 | 0.9734 |
| Gloom exfiltration pAUROC@20 | 0.8454 | 0.8285 | 0.8168 | 0.8619 |
| Test STRIDE pAUROC@20 | 0.9284 | 0.9633 | 0.9788 | 0.9665 |

**Training rows.**

| subset | rows | Qwen3.8 pAUROC@20 / AUROC | Kimi K3 pAUROC@20 / AUROC | score corr. | same side of 0.5 | label acc. Qwen / Kimi |
|---|---:|---:|---:|---:|---:|---:|
| all | 8,688 | 0.9298 / 0.9828 | 0.9636 / 0.9914 | 0.946 | 94.0% | 90.6% / 90.2% |
| strict-filtered | 7,836 | 0.9901 / 0.9979 | 1.0000 / 1.0000 | 0.969 | 96.9% | 96.9% / 100% |
| removed | 852 | 0.0019 / 0.1577 | 0 / 0 | 0.566 | 67.0% | 33.0% / 0% |

The Kimi values on the strict and removed subsets hold by construction: the
filter keeps exactly the rows where Kimi is on the label's side. On the
removed rows, AUROC covers only the three sources with both classes there
(BashArena, Gloom, STRIDE).

| pAUROC@20, all 8,688 rows | Qwen3.8 | Kimi K3 |
|---|---:|---:|
| BashArena | 0.8653 | 0.9476 |
| BashBench | 0.9705 | 0.9656 |
| CoT-Red-Handed | 0.9189 | 0.9575 |
| Gloom | 0.9387 | 0.9670 |
| STRIDE | 0.9555 | 0.9801 |

**Reading.**
- **ID.** Zero-shot, Qwen3.8 is within 0.9–2.7 points of the trained 4B
  students on macro pAUROC@20. It leads seeds 0 and 1 on Gloom and on macro
  AUROC, and trails all three on Test STRIDE.
- **Training rows.** Kimi ranks the rows better: 3.4 points of macro
  pAUROC@20, mostly BashArena (8.2) and CoT-Red-Handed (3.9). Accuracy at 0.5
  is level.
- **Removed rows.** Qwen3.8 mostly makes Kimi's errors: it lands on the
  label's side for 33% of them (BashArena 40%, Gloom 6%, STRIDE 8%).
