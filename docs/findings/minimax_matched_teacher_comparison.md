# MiniMax M3 matched full-boundary comparison

2026-09-07. Completed user-authorized diagnostic on the same 640 balanced
training samples used for Qwen/Kimi agreement. Not a held-out selection result.

## Contract and execution

`minimax/minimax-m3` via OpenRouter, pinned to `coreweave/fp4`, reasoning none,
temperature 0, max_tokens 8, top_logprobs 5, no fallbacks, no explicit caching.
Original detailed rubric and prompt hashes unchanged. Scores are normalized
over actual terminal literal 0/1 logprobs; no hard-label substitutes or fitted
corrections. All 640 unique identities, labels, source/prompt metadata, provider,
model and raw probability reconstructions passed validation. Every response
reported zero reasoning tokens and used 4 completion tokens (`Prediction:0|1`).
The first successful canary was reused; the remaining 639 requests used eight
concurrent workers. Shared client settings hashes protect cache resumption.
Run implementation commit: `9052736`; 19 relevant tests passed.

Input usage: 6,668,291 tokens; output: 2,560 tokens; cached input: 585,216
(8.776%). Reported total cost: **$1.4165173935**, including the original canary.
Cache usage occurred automatically despite no explicit cache directives.
This API diagnostic did not consume or interrupt the Lambda GPUs.

## Results

ECE is positive-class sample-weighted absolute bin gap, 10 equal-width bins.
AUROC is pooled over all 640 samples, not mean per-source AUROC.

| Teacher | AUROC | ECE | Brier | Log loss | Accuracy at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 27B FP8 | 0.949946 | 0.117087 | 0.116556 | 0.522498 | 0.857813 |
| Kimi K3 | 0.969800 | 0.044054 | 0.071498 | 0.245876 | 0.910938 |
| MiniMax M3 CoreWeave FP4 | 0.942275 | 0.078941 | 0.097374 | 0.374283 | 0.881250 |

MiniMax ECE is 0.073657 / 0.078941 / 0.083475 for 5 / 10 / 20 bins;
its intermediate calibration ranking is unchanged across these choices.
Mean predicted-class confidence is 0.940876 versus 0.881250 accuracy, a
0.059626 overconfidence gap. Mean positive probability is 0.426343 versus
the sample's constructed 0.5 label prevalence. Threshold errors: 13 false
positives, 63 false negatives (Qwen: 13/78; Kimi: 21/36).

| Source (128 each) | Qwen AUROC | Kimi AUROC | MiniMax AUROC | MiniMax ECE |
| --- | ---: | ---: | ---: | ---: |
| Bash Arena | 0.983398 | 0.993530 | 0.985107 | 0.061915 |
| Bash Bench | 0.994141 | 0.995117 | 0.988770 | 0.051383 |
| CoT Red Handed | 0.956055 | 0.992676 | 0.929443 | 0.221120 |
| Gloom | 0.972412 | 0.994873 | 0.963867 | 0.091652 |
| STRIDE | 0.983398 | 0.984375 | 0.979004 | 0.119488 |

MiniMax improves pooled calibration and proper scoring losses over Qwen but
does not improve pooled ranking on this sample. Kimi retains the best pooled
AUROC, ECE, Brier and log loss. This does not establish statistical significance
or that MiniMax would be a better prefix teacher. Source/label balancing,
imperfect labels, model-specific chat interfaces and FP4/FP8/provider differences
limit interpretation. Pooled ECE can conceal source-specific calibration errors;
the per-source results must not be replaced by the pooled number alone.
No student run, new prefix annotation, calibrator fitting or model promotion
follows from this diagnostic.

## Artifacts and reproduction

- Config: `experiments/teacher_agreement/minimax.yaml`.
- Frozen inputs: `results/teacher_agreement/minimax_m3_coreweave/manifest.json`.
- Full 640-row cache (historical filename retained):
  `results/teacher_agreement/minimax_m3_coreweave_canary.jsonl`.
- Metrics, bin counts/intervals, ties, source diagnostics and checksums:
  `results/teacher_agreement/minimax_m3_coreweave/comparison.json`.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.teacher_agreement.glm analyze \
  --config experiments/teacher_agreement/minimax.yaml
```
