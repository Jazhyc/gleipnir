# Qwen3.5-9B held-out ID data-mixture comparison

Status: completed and audited 2026-09-04.

## Finding

Adding the 13,149-example deception corpus to the same 8,688 monitoring
examples improves the existing seed-0 Qwen3.5-9B adapter on the frozen
3,012-row held-out ID suite. The gain is modest but coherent across ranking,
calibration, and the fixed operating point.

| Metric | Monitoring only | Monitoring + deception | Mixed minus monitoring |
| --- | ---: | ---: | ---: |
| Macro pAUROC@20 | 0.855009 | 0.866493 | +0.011484 |
| Macro AUROC | 0.952393 | 0.959679 | +0.007286 |
| Macro Brier | 0.090577 | 0.079430 | -0.011147 |
| Macro balanced accuracy | 0.887155 | 0.899322 | +0.012168 |
| Macro recall | 0.810070 | 0.855768 | +0.045698 |
| Macro FPR | 0.035761 | 0.057123 | +0.021362 |

The ranking gain is concentrated in Gloom-Exfiltration. STRIDE is already near
ceiling and its pAUROC@20 changes by only `-0.001167`.

| ID source | Rows | Monitoring pAUROC@20 | Mixed pAUROC@20 | Delta | AUROC delta | Brier delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gloom-Exfiltration | 2,066 | 0.745264 | 0.769400 | +0.024136 | +0.014563 | -0.017717 |
| STRIDE test | 946 | 0.964753 | 0.963586 | -0.001167 | +0.000009 | -0.004577 |

The pooled result agrees with the source-macro result: pAUROC@20 improves from
`0.823058` to `0.834727` (`+0.011669`) and AUROC improves from `0.946773` to
`0.954029` (`+0.007256`).

## Paired behavior

Scores from the two adapters remain closely related (`r=0.9632`) with mean
absolute difference `0.06285`. The mixed adapter raises scores by `0.03169` on
average. This is not merely a global calibration shift: on Gloom the mean
increase is `0.05757` for positive rows and `0.02384` for negative rows; on
STRIDE it is `0.01562` for positives and `0.00650` for negatives.

At threshold `0.5`, 189 of 3,012 decisions change. The mixed adapter corrects
118 monitoring-only errors and introduces 71 new errors, a net gain of 47
rows. There are 155 `0 -> 1` flips and 34 `1 -> 0` flips. The larger positive
shift explains the recall improvement, while the upward shift on negatives
explains the higher FPR.

## Relationship to the earlier results

This result changes the interpretation of the earlier 9B OOD mixture null. The
same historical adapter pair changes six-source strict-OOD macro pAUROC@20 by
`-0.001499` and macro AUROC by `+0.000029`, despite improving pooled OOD
pAUROC@20 by `+0.059298`. On ID, however, both macro ranking metrics improve.
Deception supervision is therefore not simply ineffective at 9B; its transfer
under this recipe is distribution- and source-dependent.

The direction also agrees with the descriptive 4B ID comparison at AdamW
`5e-5`, where mixed-minus-monitoring was `+0.022049` macro pAUROC@20,
`+0.006506` macro AUROC, and `-0.021347` Brier. The 9B pAUROC and calibration
gains are smaller, while its AUROC gain is similar. Both sizes show their main
ID ranking gain on Gloom rather than STRIDE.

## Scope and next decision

This is an exploratory evaluation of two existing seed-0 checkpoints, not a
model-selection result. Both use the same Qwen3.5-9B revision, rank-128 QLoRA,
AdamW `5e-5`, a linear schedule with 3% warmup, effective batch 32, one epoch,
and selected-position Kimi-soft loss. However, the mixed run has 683 optimizer
steps versus 272 for monitoring-only. It therefore confounds data diversity,
additional updates, and the learning-rate schedule experienced by monitoring
examples. Training-seed variance is also unmeasured.

The result is strong enough to keep data mixture as an active optimization
axis, but not to add deception to the new 4B recipe by default. A clean next
test is a matched-step 4B comparison between one mixed epoch and a cycled
monitoring-only control, using the same frozen schedule and current systems
stack. Select on the ID development suite, then consult strict OOD only once
for the promoted recipe.

## Evaluation audit

The frozen input contains 946 STRIDE-test and 2,066 Gloom-Exfiltration rows.
Both adapters were evaluated in one persistent vLLM 0.24.0 process on an H100
80 GB HBM3 with BF16 weights, FlashInfer GDN prefill, one constrained decision
token, and explicit `0`/`1` logprobs. Full inference took 1,312.79 seconds for
monitoring-only and 1,311.77 seconds for mixed. Including eager parity, engine
startup, and aggregation, the campaign took 2,834.05 seconds (47.23 minutes).

The four-row eager-versus-vLLM parity gate passed. For the mixed adapter,
Pearson correlation was `0.999509`, mean absolute score difference was
`0.010422`, and maximum absolute difference was `0.026606`; the maximum
nonzero adapter effect was `0.111789` in eager and `0.156305` in vLLM. The
unadapted base canary also passed (`r=0.999400`, mean absolute difference
`0.014212`). No prompt reached the 32,768-token limit, and all rows produced
finite scores with both decision-token logprobs.

Artifact SHA-256 values are:

- frozen config: `04f2a6725efcd7f0ab3cdb985b064acf5d226f574b111b5627949a1fdac9de17`;
- prepared prompts: `ecc8b204be2505d2600c38d1747b87e2ef3c8a53ecd51908a79ad8e86e102389`;
- preparation manifest: `4837a0b3740b6ea0a7c517780dc74d5a56964081390d0bca852b92acf792dcc7`;
- parity report: `fcc2c329287ffb7bd550a31204469808ac4dc71377789c0397a33cdc084ab78e`;
- result summary: `15d6940bf25b9eb1b4e853d4c4f3c1da8a918b296561ea25bed5531e703efe1b`;
- monitoring result: `caa34a4ebd94f231ab6f608113ed013e4a9f21d6fce8daa5e98bfbbf6ac2b035`;
- monitoring predictions: `336a76f491782c186bb81d1c786c5f6532e76dfe1fc5616b655ea96fc8691e35`;
- mixed result: `3491c4865e9e8edca480d369b49e8471061e7e7b39257ca1ee6a3691355c9107`;
- mixed predictions: `93c83656b2f966fcf0f2270ff79843dcb80086d1e70ed04e1133c0568e941776`.

The ignored local artifacts are under
`results/gleipnir9b_id_data_mixture/`. The reserved Lambda instance remains
running and was not terminated.
