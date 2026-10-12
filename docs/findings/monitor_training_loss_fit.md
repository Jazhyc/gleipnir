# Matched 4B/9B teacher fit

The complete final-checkpoint training pass shows very similar Kimi fit:
pooled teacher-to-student KL is **0.023103 for 4B** and **0.022131 for 9B**.
The 9B advantage is **4.2%**, smaller than the **13.5%** difference in the
previous average online training losses. This weakens the explanation that 4B
has a large inability to fit the supplied training targets. Generalization,
pretrained representations and domain coverage remain plausible bottlenecks;
these diagnostics do not isolate those causes or establish a capacity ceiling.

## Matched checkpoints and full training-set pass

Use the regular pair from
[student injection awareness](../../experiments/student_injection_awareness/README.md).
Both use identical 8,688 trajectories, regular student instruction and Kimi K3
probabilities: one epoch, seed 0, rank 128/alpha 256, BF16 bases/FP32 adapters,
soft BCE only, AdamW 5e-5, linear decay, 3% warmup and logical batch 32.
Both completed 272 updates. 9B uses checkpointing and has more adapter parameters;
backbone initialization and pretrained representations also differ. Historical
metadata has null Git commit fields; saved receipts and weight hashes bind the
comparison. These are the matched regular checkpoints, distinct from released
mixed-data models and the newly strictly filtered 4B checkpoint.

The [final-fit contract](../../experiments/final_training_fit/README.md) scores
all original training prompts with each fixed final adapter. It preserves
original unfiltered targets and labels, including teacher/source disagreements.
Merged BF16 backbones and unchanged weight rows projected in FP32 are matched
across sizes, with TF32 disabled. Each size passes master/merge and serving
parity before full scoring. This is an explicit diagnostic exception to the
optimized 4B-only serving default, with fresh validation of the 9B geometry.
No retraining, held-out scoring, checkpoint selection or promotion occurred.

Losses use original teacher probability q and raw student logit margin m:
soft CE = log(1 + exp(m)) - q*m, evaluated stably; KL subtracts Bernoulli
teacher entropy. This avoids clipping saturated probabilities. Pooled entropy
is **0.156991554 nats** for both models. Unlike online epoch summaries, these
are exact row means for a fixed checkpoint over the full population.

| Final-checkpoint pooled metric | 4B | 9B |
|---|---:|---:|
| Soft cross-entropy (nats) | 0.180095 | 0.179123 |
| Teacher-to-student KL (nats) | 0.023103 | 0.022131 |
| Mean absolute probability error | 0.036373 | 0.036102 |

The pooled KL gap is just **0.000972 nats**; probability MAE differs by
0.000271. Equal-source macro KL is 0.024602 versus 0.023585. Macro probability
MAE is 0.036570 versus 0.036660, slightly worse for 9B despite its lower KL.

| Training source | Rows | 4B KL | 9B KL |
|---|---:|---:|---:|
| Bash Arena | 2,790 | 0.026013 | 0.025838 |
| Bash Bench | 1,436 | 0.020951 | 0.016576 |
| CoT Red-Handed | 2,340 | 0.015778 | 0.015235 |
| Gloom | 772 | 0.034991 | 0.035701 |
| Stride | 1,350 | 0.025279 | 0.024573 |

Most of the pooled advantage comes from Bash Bench; 9B has slightly worse KL
on Gloom. Full source/label breakdowns remain in each model's `fit.json`.
Descriptive audit quantiles show median KL 0.003620 versus 0.003135, but
95th-percentile KL 0.095450 versus 0.099533. Small averages therefore do not
mean all training examples are fitted well, and quantile improvements are uneven.
The result supports a small recipe-specific fitting advantage for 9B, with
substantial similarity elsewhere. It does not show that either model converged,
that fitting training examples implies transferable understanding, or that
rare hard examples are represented adequately. The small point-estimate gap
also lacks seed/repeated-serving uncertainty; canary parity qualifies the
merged approximation rather than proving full-population master equivalence.

## Historical online loss comparison

| Online statistic (nats) | 4B | 9B |
|---|---:|---:|
| Trainer mean CE over the epoch | 0.233122 | 0.222823 |
| Approximate mean KL | 0.076130 | 0.065831 |
| Mean CE over last 50 logged updates (221–270) | 0.204230 | 0.196459 |
| Last five-update logged CE (266–270) | 0.197231 | 0.188655 |

Online losses come from changing checkpoints before updates; the last two
updates have no separate report. Epoch KL subtracts whole-population entropy
from Trainer's update mean and is approximate because the final batch contains
16 rows. Do not use whole-population entropy for late-window KL. The larger
online gap, which narrows in the fixed final pass, is consistent with a learning
speed or initialization advantage; it does not prove that explanation.
The [original curve](../../figures/monitor_training_losses.png) remains an
online-loss plot. Regenerate with
`PYTHONPATH=src:. .venv/bin/python scripts/plot_monitor_training_losses.py`;
its provenance binds metadata, teacher targets, source and exports.

## Historical multi-epoch 4B losses

Saved soft-only runs supply an exposure diagnostic. In the
[five-epoch 20% subset run](../../experiments/monitoring_subset_duration/README.md),
all epochs reuse the same 1,738 parents and fixed Kimi targets, with fresh AdamW
2e-5 training, rank-128 NF4 QLoRA/BF16 and a five-epoch linear schedule.
There are no MIL, prefix, completion or hard-label auxiliary terms.

| Epoch within that five-epoch run | Mean logged online CE |
|---|---:|
| 1 | 0.339623 |
| 2 | 0.200761 |
| 3 | 0.181323 |
| 4 | 0.167294 |
| 5 | 0.159901 |

These are 11 five-update reports per epoch, not fixed-checkpoint losses.
The selected targets have row-mean entropy **0.151576457 nats**; that is a
reference, not an exact entropy correction for differently weighted log windows.
Repeated exposure continues to reduce the training objective well after one
epoch under this recipe. This supports removable optimization/exposure error,
without establishing the final adapter's exact training KL or an OOD gain.

The [full-data two-epoch 2e-5 run](../../experiments/monitoring_duration/README.md)
also falls from 0.239100 to 0.180161 in mean logged CE grouped by reported epoch.
Five-update windows can straddle its epoch boundary. Its historical ID pAUROC
improves from the separate one-epoch endpoint's 0.871066 to 0.882952; at 1e-5,
two epochs instead worsen ID ranking. On the 20% subset, soft-only historical
ID pAUROC at 1/2/3/5 epochs is 0.886670/0.875770/0.886438/0.890454.
Loss reduction therefore does not imply monotonic held-out improvement.

These historical NF4/2e-5 unfiltered runs differ from the recent BF16/5e-5
one-epoch pair in precision, schedule, and sometimes training population.
Historical ID retains reasoning, unlike the current CoT-removed contract.
No matched OOD scoring was completed for these soft-only duration cells; the
separate three-epoch MIL OOD result cannot fill that gap. The local audit under
`results/training_loss_capacity_audit/multi_epoch.json` binds the selected
parent list, metadata hashes, objective weights and per-epoch log extraction.
No new inference, training or lifecycle operation was performed for this review.

## Completion and independent audit

`final-training-fit02` completed at **03:14:02 CEST on 2026-10-12**, 43m07s
from the driver's recorded start on the existing NC2 B200. Implementation is `e2f494f`;
10 focused tests and lint passed. All **17,376 predictions** cover matched IDs,
rendered prompts and token lengths. Each model scores **83,816,369 input tokens**,
with no truncation. Local `results/final-training-fit02/completion_audit.json`
independently reconstructs every prompt from pinned tokenizer assets, checks
original teacher/label identities, raw responses and prediction hashes,
recomputes all CE/entropy/KL/MAE source/label and macro metrics, and checks both
parity gates and actual BF16/FP32-head dispatch. Token counts are verified
pairwise and against the recorded full training-token total, without another
local retokenization pass.

| Model | Master/merge MAE / correlation | Master/serving MAE / correlation | Scoring seconds | Input tokens/s |
|---|---|---|---:|---:|
| 4B | 0.002743 / 0.999866 | 0.003898 / 0.999892 | 749.57 | 111,819 |
| 9B | 0.002148 / 0.999870 | 0.002506 / 0.999942 | 1,137.53 | 73,683 |

Requests/s are 11.59 and 7.64 respectively; raw per-request latencies remain
saved. These are single-pass timing receipts, not repeated throughput estimates.
Prediction SHA256 values:

- 4B: `62b880f316aa9ba6e88fc2bbf1fa7c00ca68e0003e3ab16ab304c2adc89a3c23`
- 9B: `7bac660c48e8e17205e7ced91c0644fd74df956f050e4e89c70fb2b9330202e6`

Frozen contracts, normalized receipts, raw predictions and logs were collected
locally without model weights. Preserved preparation/schema failures and the
receipt-only compatibility fix are in the experiment record. Scorer shutdown
logged an EngineDeadError for 4B and semaphore cleanup warnings after complete
scoring; there were no excluded rows or scoring failures. The subsequent live
GPU check shows 0% utilization and 0 MiB allocated. Capacity remains running,
original adapters/caches remain intact, and only the final 9B merge occupies
the shared disposable slot.
