# Optimized-stack drift on the last full-trained adapter

2026-10-08. The initial grouped evaluation of the last full-trained 4B adapter on the
optimized vLLM stack shows a material ID regression against the same adapter's
archived BF16 dynamic-LoRA predictions. Source-macro AUROC falls 1.27 percentage
points and raw normalized pAUROC@20 falls 3.48 points. Gloom accounts for most
of the loss. Reproducing the accepted small canary does not establish held-out
quality parity. Keep the user-selected serving default and frozen controls;
this measurement neither promotes a new checkpoint nor isolates a kernel cause.
The continuous-admission follow-up below gives a small throughput gain with
longer request latency and retains the quality regression.

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

## Continuous-admission follow-up

The user requested continuous admission on 2026-10-08. Refill request slots as
responses complete, reuse one HTTP pool and persist results through an
asynchronous journal. The [experiment contract](../../experiments/b200_optimized_id/README.md#continuous-admission-comparison)
defines input/adapter-bound recovery, durability and failure handling.

Freeze the grouped `id02` artifacts in `continuous.json`, use one new full pass
(`continuous01`), and leave the historical grouped and BF16 controls unchanged.
The exact server command, pooling correction, rendered workload, merged model
and selected backend match the grouped control. The same GPU UUID is verified
live; reuse the persistent compiler cache and reproduce the same accepted real
canary. Startup takes 106.380 seconds and is excluded, as is quick64 warmup.
This is a sequential, single-pass client comparison with a server restart and
persistent cache reuse. It combines admission, HTTP-pool reuse and persistence;
new batch combinations can encounter first-use work. It does not isolate the
barrier's causal cost or establish a precise speed or quality repeat distribution.

| Measurement | Grouped `id02` | Continuous `continuous01` |
| --- | ---: | ---: |
| Timed full-pass duration | 153.131 s | 149.063 s |
| Input tokens/s | 220,406 | 226,420 |
| Requests/s | 19.67 | 20.21 |
| Request latency p50 | 2.349 s | 7.549 s |
| Request latency p95 | 7.737 s | 9.348 s |
| Request latency p99 | 9.057 s | 9.641 s |
| Mean HTTP requests in flight | 66.06 | 124.18 |
| Source-macro AUROC | 0.953200 | 0.952724 |
| Source-macro pAUROC@20 | 0.851263 | 0.850323 |

Measured throughput increases **2.73%**, elapsed time decreases **2.66%**, and
p50 latency increases **3.21x**. Mean HTTP concurrency is the integral of request
durations divided by full-pass time, not GPU occupancy. Both clients cap at
128; continuous admission fills those slots much more consistently. Most of
that increase manifests as request waiting rather than a large throughput gain
in this trial. The grouping barrier is therefore not evidenced as a large
throughput bottleneck by this comparison; the small speed difference remains
subject to unmeasured run variation.

The asynchronous writer spends 26.253 seconds in overlapping append/flush/fsync
I/O, with a peak queue of 29 records versus capacity 256. Aggregate queue-put
wait/overhead is 0.0087 seconds; the queue never fills. Persistence is not holding
up admission. Final drain and HTTP cleanup take 0.0261 seconds after the last
validated response; ordered prediction export adds 0.0649 seconds. These timings
include stronger fsync durability than the original grouped writes and should
not be interpreted as 26 seconds of serialized inference overhead. The grouped
timer includes batch checkpoints and excludes its final ordered export; the
continuous duration includes both the journal and its 0.0649-second export.

Compared with grouped optimized scores, MAE is 0.015495, Pearson 0.995214,
absolute-error p95 0.079704 and maximum 0.358357. There are **44 threshold
disagreements** (16 negative-to-positive, 28 positive-to-negative), 1.46% of
inputs. Source-macro AUROC changes -0.0475 percentage points and pAUROC@20
-0.0940 points. Gloom AUROC/pAUROC@20 change -0.1002/-0.3992 points; STRIDE
+0.0052/+0.2111 points. Pooled AUROC changes -0.0415 points and Brier increases
0.000653. Source-macro Brier is 0.117187 versus 0.116865. Keep this schedule
sensitivity visible rather than treating client batching as numerically neutral.

Relative to BF16, continuous macro AUROC is lower by 1.3202 points and
pAUROC@20 by 3.5767 points. Score MAE is 0.071444, Pearson 0.960388 and threshold
disagreements 217. Pooled Brier/ECE are 0.130876/0.152391. The original
whole-stack quality regression remains; this client change does not resolve it.
No precision, threshold, checkpoint or quality-default promotion occurs.

All 42 collected remote artifacts match sizes and SHA-256 values in
`results/b200_optimized_id/continuous01_remote_artifacts.json`. The local
journal audit validates every persisted response against exact input identities,
raw two-logit consistency and frozen contract bindings, verifies equality to the
ordered predictions, preserves input dispatch order and observes peak HTTP
concurrency exactly 128. Raw start/completion offsets allow later request-trace
analysis. Post-run sources, paired comparisons against both controls, plots and
local audits are saved separately from immutable execution snapshots.

Eleven focused local mocked tests pass; the four admission/recovery tests also
pass in the actual serving runtime. Local tests require running outside the
restricted sandbox: a minimal `asyncio.to_thread` reproduction also times out
inside it, while unrestricted tests and remote thread wakeups work. The only
server is retired after the pass; closure shows no GPU processes and 0 MiB.
The existing Pod, checkpoints, shared caches and previous results are preserved.
