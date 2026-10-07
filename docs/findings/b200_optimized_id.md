# Optimized-stack drift on the last full-trained adapter

2026-10-08. One complete evaluation of the last full-trained 4B adapter on the
optimized vLLM stack shows a material ID regression against the same adapter's
archived BF16 dynamic-LoRA predictions. Source-macro AUROC falls 1.27 percentage
points and raw normalized pAUROC@20 falls 3.48 points. Gloom accounts for most
of the loss. Reproducing the accepted small canary does not establish held-out
quality parity. Keep the user-selected serving default and frozen controls;
this measurement neither promotes a new checkpoint nor isolates a kernel cause.

## Frozen comparison

The selected adapter is the completed full-epoch FP4/FA4 training replication,
272 updates, FP32 causal master `8dbc1a2e...`, rebased serving artifact
`d13be8b2...`. Later twenty-update systems adapters are scratch controls. Reuse
its source-bound FP32-accumulated/BF16-exported merged checkpoint and the same
3,012 canonical CoT-removed ID examples: 946 STRIDE and 2,066 Gloom. All ordered
IDs, labels, source lineage, source/rendered prompt hashes and tokenizer counts
match the original predictions. There are 33,750,959 prompt tokens, mean
11,205.5, maximum 29,513. No truncation, threshold fitting, training, ID tuning,
OOD evaluation or quality promotion. See the
[experiment contract](../../experiments/b200_optimized_id/README.md).

Use the selected native Gigatoken/direct FROST scorer with FP4 MLPs and large
GDN/full-attention projections, direct FP4 SwiGLU output, MXFP8 full-attention
prefill, BF16 operands/cache and FP32 gates/state. Causal LAST two-row scoring,
prefix caching off, synchronous FCFS, 32K context/chunk budget, 128 engine
sequences and the source-bound pooling reservation correction remain fixed.
The old BF16 pass used dynamic LoRA, prefix caching on, 16 engine sequences,
15% GPU utilization alongside an idle trainer, and another B200 host. This is
same-adapter **whole-stack drift**, not isolated quantization or a matched
causal speed comparison.

The user reduced the plan to one full pass while the first client was active.
Preserve its 512 completed rows as an excluded incomplete attempt (`id01`).
Replace only that client and reuse the independently resident server for the
single complete pass (`id02`); no restart and no second/third complete pass.
Exclude the training-source quick64 warmup. Some initial ID shapes were already
used by the partial attempt; the remaining shapes are encountered naturally.
This is neither a cold-start nor a uniformly warmed measurement. Repeat
variation is unmeasured.

The 20-example canary reproduces the previously accepted optimized recipe:
score MAE 0.000807, Pearson 0.999967, nonzero adapter effect. Against the separate
frozen BF16 causal-master canary, MAE is 0.019734 and Pearson 0.996962. Retain
the inherited failed strict master check separately from finite acceptance.

## Ranking, calibration and scores

Metrics use the original labels and raw normalized pAUROC@20 definition.
Recomputed pooled/source-macro BF16 metrics match the archived result to
1e-12. One complete paired observation per input:

| Population | BF16 AUROC | Optimized AUROC | Change, pp | BF16 pAUROC@20 | Optimized pAUROC@20 | Change, pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Source macro | 0.965927 | 0.953200 | -1.273 | 0.886090 | 0.851263 | -3.483 |
| Pooled | 0.960818 | 0.947099 | -1.372 | 0.866566 | 0.824900 | -4.167 |
| Gloom | 0.943579 | 0.919426 | -2.415 | 0.825344 | 0.761825 | -6.352 |
| STRIDE | 0.988275 | 0.986974 | -0.130 | 0.946835 | 0.940701 | -0.613 |

Source-macro Brier worsens from 0.086493 to 0.116865. Pooled Brier rises from
0.094087 to 0.130223, ten-bin ECE from 0.096407 to 0.151292 and log loss from
0.308478 to 0.408931. At the unchanged 0.5 threshold, pooled recall falls from
79.29% to 68.59% and FPR from 2.78% to 1.99%. Gloom recall falls from 74.78% to
60.62%; STRIDE recall from 87.35% to 82.84%.

Mean signed score difference is -0.054885, MAE 0.070682, absolute-error p50
0.029724, p95 0.288323, p99 0.433557 and maximum 0.684610. Pearson is 0.960791;
Spearman 0.972165. There are 213 threshold flips (7.07%): 198 positive-to-negative
and 15 negative-to-positive. Gloom has 177 flips, STRIDE 36. Margin MAE is
0.575822 with signed mean -0.396539. Preserve per-example scores, margins,
latencies and exact identities rather than claiming quality parity from the
mean or small canary.

Exact unique-score counts are 1,472 BF16 and 83 optimized; with scores rounded
to six decimals they are 86 and 83. The archived logprob-subtraction path has
small floating-point differences around otherwise tied margins. Exact-count
contrast therefore overstates the change in effective score resolution. Rounding
BF16 scores to six decimals changes macro AUROC by only +0.000049 and pAUROC@20
by +0.000040, so this does not explain the observed regression.
The analysis also preserves per-source metrics, calibration bins, margin-error
percentiles, score correlations, fixed-threshold flips and unmeasured-repeat
flags. `score_drift.png`/`.svg` plot paired scores and signed error by source.

## Throughput and recovery groups

Keep the original 128-row partitions and at most 128 concurrent HTTP requests
within each partition. The next group waits for every request in the current
group. A group can therefore drain to a few long requests, leaving admission
slots unfilled. Checkpoint recovery does not inherently require this barrier:
completed responses could be persisted during continuous request admission.
That is a possible subsequent experiment, not an intervention in this run.

The complete pass takes **153.131 seconds**, **220,406 input tokens/s** and
**19.67 requests/s**. Request latency p50/p95/p99 is **2.349/7.737/9.057 seconds**,
including localhost HTTP, text encoding, engine waiting and inference, excluding
semaphore waiting. This workload is substantially longer than the earlier
full320 systems cohort. The archived BF16 pass takes 665.178 seconds; the
4.34x elapsed-time ratio describes the combined changed setup, not a matched
precision speedup.

Summed HTTP trial time is 151.002 seconds. The 24 checkpoint writes take
1.783 seconds (**1.16%** of pass time); total non-HTTP pass overhead is 2.129
seconds. Saving groups is a small direct cost. Summed within-group max-minus-p90
request-latency gaps are 12.706 seconds (8.30% of pass time), and max-minus-p50
gaps are 75.695 seconds. These spreads suggest straggler tails but are not
measured lost GPU time or predicted continuous-admission gains. Request start
times can differ, so archived `completion_*` fields (computed from request
latencies) are proxies, not actual batch-relative completion timestamps. There
is no matched barrier-free trial or GPU occupancy trace to identify the causal
cost. New runs name these fields `request_latency_*` explicitly.

## Artifacts and closure

`results/b200_optimized_id/id02/` contains the complete 24 batch checkpoints,
paired predictions, frozen workload/reference, selection/merge/source bindings,
canary/native/compiler/precision receipts, measurements and retirement.
`id01/` retains the superseded source/settings contract and partial batches.
`results/b200_optimized_id/remote_artifacts.json` binds all 87 collected remote
artifacts; all sizes and SHA-256 values verify. Post-run analysis and plot
sources are saved separately from immutable executed-source snapshots.

Six focused identity, label/order, repeat and live-command tests pass; Ruff is
clean. The verified API/engine are retired after the single full pass. Closure
shows 0 MiB GPU allocation and no GPU processes. The existing Pod, FP32 master,
merged checkpoint, network volume, caches and historical evidence remain intact.
