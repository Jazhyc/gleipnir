# Qwen3.8-Flash matched direct-logprob comparison

2026-09-07. Complete on 640 source/label-balanced training trajectories.
Not a held-out promotion result. No additional teacher models were called.

## Interface failure and recovery

Alibaba is the sole catalogued OpenRouter endpoint for `qwen/qwen3.8-flash`;
quantization is unspecified. The original eight-token prediction-line canary
passed, but the larger pass emitted free-form explanations on multiple rows
and encountered upstream 429s. Stopped PID 1670236 (exit 143); preserved 133
successful cached rows and recorded cancellation in the original result root.
No metric is reported on that selected successful subset.

A new, separately frozen version retains the original user rubric and adds
assistant prefill `Prediction:` with `partial: true`. Generate one token,
reasoning none, temperature zero, and request actual top-5 logprobs for scalar
0/1 extraction. This explicitly changes chat serialization, analogous to the
local Qwen prefilled boundary, rather than letting the model generate an
explanation. Old scores are not mixed into the new comparison. Prefill content
and partial mode are included in the request-settings hash.

Ten source/label canaries passed before the full run. The full pass at
concurrency four completed 637 rows; three upstream HTTP-429 failures completed
on an unchanged-settings concurrency-one resume. All 640 unique identities,
labels, source metadata, original prompt hashes, model/provider identities and
raw 0/1 score reconstructions passed validation. All responses report exactly
one completion token and zero reasoning tokens. No imputation, target fitting,
new student/prefix/GPU work or ID/OOD selection was performed.
Implementation commit: `1e003b7`; 22 client/teacher tests passed.

## Pooled results

ECE is positive-class weighted absolute gap in 10 equal-width probability
bins. AUROC is pooled over all 640 samples, not an average of source AUROCs.

| Teacher | AUROC | ECE | Brier | Log loss |
| --- | ---: | ---: | ---: | ---: |
| Kimi K3 | 0.969800 | 0.044054 | 0.071498 | 0.245876 |
| Kimi K2.6 Inceptron INT4 | 0.960449 | 0.051102 | 0.081665 | 0.270839 |
| Qwen 3.5 27B FP8 | 0.949946 | 0.117087 | 0.116556 | 0.522498 |
| MiniMax M3 CoreWeave FP4 | 0.942275 | 0.078941 | 0.097374 | 0.374283 |
| Qwen3.8-Flash Alibaba, prefilled | 0.933633 | 0.090544 | 0.119205 | 0.401888 |

Qwen3.8-Flash ECE at 5/10/20 bins: 0.085339 / 0.090544 / 0.091132.
Accuracy at 0.5 is 0.846875; mean confidence 0.919592; confidence-minus-accuracy
0.072717. Mean positive probability 0.422417 versus constructed prevalence 0.5.
At threshold 0.5: 21 false positives, 77 false negatives; 582 distinct scores.

| Source (128 each) | Qwen3.8 AUROC | ECE | Brier |
| --- | ---: | ---: | ---: |
| Bash Arena | 0.967529 | 0.108832 | 0.094388 |
| Bash Bench | 0.998413 | 0.079453 | 0.043584 |
| CoT Red Handed | 0.971436 | 0.225457 | 0.197071 |
| Gloom | 0.994385 | 0.232427 | 0.165014 |
| STRIDE | 0.975830 | 0.111863 | 0.095970 |

The tested prefilled endpoint does not improve the pooled ranking/calibration
tradeoff over K2.6. However, within-source AUROC is often high: Gloom in
particular combines 0.994385 ranking with 0.232427 calibration error. Thus
poor pooled ranking should not be described as uniformly poor discrimination
within every source; cross-source score scales contribute to the diagnostic.
ECE/Brier need not order models identically. No statistical significance,
prefix-teacher quality or student-transfer conclusion is established. Original
label quality, balanced source mixture, provider/quantization and differing
assistant serialization remain limitations.

## Cost and artifacts

Valid full pass: 7,037,644 input tokens, 640 output tokens, 3,155,712 cached
input tokens (44.8405%), reported cost **$0.62675117208**.
Ten prefill canaries: $0.0108734472. The stopped prediction-line pass retains
133 successful rows costing $0.35856586656. Sum of these recorded successful
charges is **$0.99619048584**; malformed or interrupted calls may add charges
not represented by successful-row caches. Cache hits were automatic; none
were assumed in the budget.

- Config: `experiments/teacher_agreement/qwen38flash_prefill.yaml`.
- Complete root: `results/teacher_agreement/qwen38flash_alibaba_prefill/`.
- `manifest.json` freezes original inputs, configuration and additional baselines.
- `scores.jsonl` retains all 640 raw alternatives, normalized scores and usage.
- `comparison.json` includes all five pooled/source metrics and artifact hashes.
- Original stopped root: `results/teacher_agreement/qwen38flash_alibaba/`.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.teacher_agreement.glm analyze \
  --config experiments/teacher_agreement/qwen38flash_prefill.yaml
```
