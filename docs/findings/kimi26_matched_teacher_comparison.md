# Kimi K2.6 matched training-population diagnostic

2026-09-07. Complete on all 640 original source/label-balanced full trajectories.
This is not a held-out model-selection result and does not use OOD scores.

## Interface and coverage

Requested `moonshotai/kimi-k2.6`, pinned to Inceptron `inceptron/int4`;
temperature zero, reasoning none, max_tokens 8, actual top-5 logprobs at the
terminal literal 0/1 decision. The original detailed rubric and prompt hashes
are unchanged. All 640 unique IDs, labels, source metadata, prompt hashes,
model/provider identities and score reconstructions passed validation. Every
accepted row reports zero reasoning tokens. The comparison also revalidates
the existing MiniMax cache and freezes its checksum; no teacher scores are fitted
or altered. Shared client resumption checks request-setting hashes.

Initial concurrency eight completed 634 rows; six exhausted bounded retries
on upstream shared-pool HTTP 429 responses. A transport-only concurrency-two
resume completed precisely those six rows, with the same request settings and
no provider fallback. No GLM, student, prefix or GPU jobs were launched.
Initial implementation commit: `0154649`; 19 relevant tests passed before
launch, then 12 client/teacher tests passed after the concurrency override.

Input: 6,505,168 tokens; output: 2,560; cached input: 606,304 (9.320%).
Reported successful-row charges: **$3.3809627808**, including the initial
canary; rejected requests are not included in this successful-row sum.

## Pooled comparison

AUROC uses all 640 samples jointly. ECE is positive-class calibration error
with 10 equal-width bins, weighted by bin counts. Brier and log loss are
proper scoring losses, not pure measures of calibration.

| Teacher | AUROC | ECE | Brier | Log loss | Accuracy at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kimi K3 | 0.969800 | 0.044054 | 0.071498 | 0.245876 | 0.910938 |
| Kimi K2.6 Inceptron INT4 | 0.960449 | 0.051102 | 0.081665 | 0.270839 | 0.892188 |
| Qwen 3.5 27B FP8 | 0.949946 | 0.117087 | 0.116556 | 0.522498 | 0.857813 |
| MiniMax M3 CoreWeave FP4 | 0.942275 | 0.078941 | 0.097374 | 0.374283 | 0.881250 |

K2.6 ECE at 5/10/20 bins: 0.040780 / 0.051102 / 0.058190. Its position
between K3 and the other two teachers is unchanged across these bin choices.
Mean predicted-class confidence 0.925847 versus accuracy 0.892188 gives an
overconfidence gap of 0.033659. Mean positive probability is 0.465440 against
the constructed prevalence 0.5. At threshold 0.5: 22 false positives and 47
false negatives. There are 582 distinct numerical scores among 640 samples.

| Source (128 each) | K2.6 AUROC | K2.6 ECE | K2.6 Brier |
| --- | ---: | ---: | ---: |
| Bash Arena | 0.976563 | 0.130355 | 0.079977 |
| Bash Bench | 0.999512 | 0.059698 | 0.036165 |
| CoT Red Handed | 0.937988 | 0.181047 | 0.163249 |
| Gloom | 0.998779 | 0.100234 | 0.060024 |
| STRIDE | 0.984375 | 0.066648 | 0.068912 |

K2.6 is the strongest cheaper candidate measured here on both pooled AUROC
and calibration point estimates, but does not surpass K3 overall. Source
results vary: e.g. K2.6 Gloom AUROC exceeds K3 while its Gloom calibration is
worse. Do not infer statistical significance, intermediate-prefix quality,
student gains or universal calibration from this balanced endpoint sample.
Quantization, provider/chat serialization, imperfect labels, and the changed
source mixture remain caveats. Pooled ECE can conceal source-level errors.
No target rescaling, student training or promotion is implied.

## Artifacts

Config: `experiments/teacher_agreement/kimi26.yaml`.
Root: `results/teacher_agreement/kimi26_inceptron/`; `manifest.json` freezes
inputs, `canary.jsonl` holds the complete 640-row resumable score cache, and
`comparison.json` records all four models' pooled/source metrics, bins and hashes.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.teacher_agreement.glm analyze \
  --config experiments/teacher_agreement/kimi26.yaml
```
