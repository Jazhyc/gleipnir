# Gleipnir 4B held-out ID baseline

Status: completed and audited 2026-09-03.

## Question and frozen design

Before selecting new training hyperparameters, evaluate the completed mixed-data
Gleipnir 4B adapter and its unadapted Qwen3.5-4B backbone on the exact held-out
in-distribution population. The suite contains 3,012 trajectories: 2,066 from
Gloom-Exfiltration and 946 from STRIDE test. These sources occur in training,
but the evaluation trajectories are held out. This suite is the development
comparison set for subsequent ablations, not the final strict-OOD test.

Both conditions use the frozen compact student prompt, non-thinking assistant
boundary, and normalized next-token scores for literal `0` and `1`. The primary
metric is the unweighted two-source macro pAUROC@20. The executable config is
[`../../experiments/tool_trajectory_monitoring/gleipnir4b_id_benchmark.json`](../../experiments/tool_trajectory_monitoring/gleipnir4b_id_benchmark.json),
with SHA-256
`7b523c5341882c8c66016ce7301cf3d017bdec1893afa39f3f5a1a072cd2758c`.

## Result

| Condition | Macro pAUROC@20 | Macro AUROC | Macro Brier | Balanced accuracy | Recall | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B base | 0.604755 | 0.860769 | 0.231949 | 0.697290 | 0.429187 | 0.034606 |
| Gleipnir 4B | 0.869194 | 0.957679 | 0.074303 | 0.905482 | 0.850790 | 0.039826 |
| Gleipnir minus base | +0.264439 | +0.096910 | -0.157646 | +0.208192 | +0.421604 | +0.005220 |

The adapter improves the frozen primary metric on both sources rather than only
on the larger source.

| Source | Base pAUROC@20 | Gleipnir pAUROC@20 | Base AUROC | Gleipnir AUROC | Base Brier | Gleipnir Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gloom-Exfiltration | 0.420869 | 0.772335 | 0.774628 | 0.922492 | 0.251328 | 0.116677 |
| STRIDE test | 0.788641 | 0.966052 | 0.946910 | 0.992866 | 0.212570 | 0.031929 |

At the fixed 0.5 threshold, Gloom recall rises from `0.329777` to `0.744908`
while FPR rises from `0.050242` to `0.057971`. STRIDE recall rises from
`0.528596` to `0.956672` while FPR rises from `0.018970` to `0.021680`.
Pooled pAUROC@20 rises from `0.510935` to `0.847502`, and pooled AUROC from
`0.831698` to `0.954270`. The adapter produces 1,631 unique scores versus 1,173
for the base; tied rows fall from 1,839 to 1,381.

## Serving parity and backend choice

The balanced four-row master-versus-serving FlashInfer gate passed. Base scores
had Pearson correlation `0.999716` and mean absolute difference `0.014076`;
adapter scores had correlation effectively `1.0` and mean absolute difference
`1.53e-8`. The served adapter's maximum effect relative to base was `0.462117`.

A matched Triton systems canary initialized in `119.80` seconds versus
`364.08` seconds for uncached FlashInfer. Triton's adapter scores were stable,
but the unadapted base narrowly failed the frozen eager-parity mean-error limit:
`0.020512 > 0.020000`, with maximum difference `0.054430`. FlashInfer was
therefore retained for the official result. This is a numerical-contract
failure, not evidence of a crash or general Triton reliability problem.

## Runtime and artifacts

The official persistent-engine run used vLLM 0.24.0, PyTorch 2.11.0+cu130, and
one NVIDIA H100 80 GB HBM3 on `gleipnir-improvement`. It processed 34,631,573
prompt tokens per condition, with mean length 11,497.87 and maximum 29,513.
The resumed base scoring pass took 526.30 seconds and the adapter pass took
861.24 seconds. End-to-end wall time was about 1,416 seconds (23m36s), including
an 11.07-second cached engine start, so startup was 0.8% of this run.

Replacing that cached start with the observed cold starts projects 29m29s for
FlashInfer (20.6% startup) and 25m25s for Triton (7.9% startup), assuming equal
full-run throughput. Triton would save about 244 seconds on a fresh node, but
not enough to justify changing this result's backend after its parity failure.
On the long-lived Lambda target, the populated FlashInfer cache makes startup
negligible.

All 3,012 identities are unique and complete in both conditions, all scores are
finite, and the maximum prompt stays below 32,768 tokens. SHA-256 values are:

- base result: `47be338cb4ecc2da9864879dd9d59dc3dad21ab0ce4028d95f6ba9beb65875c5`;
- base predictions: `9e60bf9159af001526ffc86b4799f8bbae15da8820c9a99ec87871a9a7e7b000`;
- adapter result: `2c3620805133908d56348a67e1d27d70377ef764e488c9268f2fce8194dac0ba`;
- adapter predictions: `f62ec78921b008080702f648efd90da0e3d06dd48e3b425eebca1ec23bf5bbde`;
- serving-parity report: `f785ed22291d1f0e793b69e6595a0822b05808d024c27a83b11155d32076a42f`.

## Interpretation

The completed Gleipnir 4B checkpoint is a strong ID baseline and materially
improves ranking, calibration, and fixed-threshold recall over the unadapted
backbone on both source families. Subsequent optimizer and hyperparameter
ablations should compare against this frozen result and use this suite only for
selection. The strict six-source OOD suite remains untouched by that selection
and is still required to establish generalization.
