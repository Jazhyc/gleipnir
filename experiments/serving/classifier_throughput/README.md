# Direct-classifier inference throughput

This experiment measures how much of the Qwen3.5-9B monitor's direct-boundary
inference time can be recovered through serving configuration alone, before
changing precision or comparing against autoregressive reasoning.

## Hypothesis and intervention

The current offline vLLM path uses balanced scheduling defaults and leaves the
progress renderer enabled. Because each request performs a substantial prefill
and produces one constrained token, disabling progress overhead and selecting
vLLM's throughput mode with a larger scheduled-token budget should improve
aggregate examples per second without changing direct scores.

The first sweep compares four BF16 dynamic-LoRA conditions on one GPU:

1. `balanced_current`: the existing balanced scheduler and progress behavior;
2. `balanced_no_tqdm`: only remove progress rendering;
3. `throughput_default`: vLLM throughput mode and its hardware default budget;
4. `throughput_32768`: throughput mode with 32,768 batched prompt tokens.

After completing that fixed scheduler sweep, `merged_bf16` tests the next
predeclared intervention: safe-merge the same adapter into the pinned BF16 base
weights and rerun the selected balanced scheduler without vLLM's dynamic-LoRA
path. The hypothesis is that removing per-layer adapter dispatch and separate
low-rank projections can produce a material gain when scheduling changes do not.
The merged artifact remains an ignored, reconstructable result whose source
revision, adapter checksum, file checksums, dtype, and merge mode are recorded.

Each condition loads the same Qwen3.5-9B revision and Phoenix 8.1 rank-16 BF16
serving adapter, performs an untimed 64-row warmup, resets the prefix cache, and
runs three timed passes. The cache is reset between repeats so identical prompt
replays cannot turn into full-prefix cache hits. The benchmark records resolved
engine settings, source and artifact hashes, hardware/software versions,
initialization time, rows/s, prompt tokens/s, direct scores, and monitor metrics.
Because vLLM may use numerically nondeterministic scheduling or kernels even at
temperature zero, retain all repeat score arrays and report their per-row range
and threshold instability. Use each row's median score for cross-condition
parity; do not discard a throughput measurement merely because repeated BF16
scores are not bit-identical.

## Baseline, data, and parity rule

`balanced_current` is the baseline. The workload is the frozen 822-row
competition validation set already used by the data- and adapter-scaling
experiments. It is suitable for selecting a systems configuration for this
known prompt distribution; it is not evidence that the speedup transfers to
other prompt lengths, GPU families, online arrival patterns, or multimodal
inputs.

All conditions use the same input order, one-token constrained decoding, and
dynamic LoRA serving path. A condition is eligible for promotion only if its
maximum absolute score change from the baseline is at most `1e-4` and it causes
no prediction flips at the fixed 0.5 threshold. Report metric and ordering
changes even when this gate passes.

## Stop condition and next steps

Complete the predeclared four-condition sweep once, then the merged-BF16
condition once. Do not tune the token budget
against individual datasets or quality metrics. Reject a condition on OOM,
missing scores, prompt-order mismatch, or parity failure, and report any
within-condition threshold instability. Select the fastest passing condition by
median rows/s. Only then benchmark FP8 with a new parity gate and a packed
Transformers final-position forward, in that order. A100 results select
configurations to retest on H100; they are not an H100 throughput estimate.

## Scheduler sweep finding (2026-08-12)

Slurm job `30597418` completed the scheduler sweep on one A100. The current
balanced configuration measured 34.06 rows/s (10.60k prompt tokens/s). Removing
the progress renderer measured 33.91 rows/s, throughput mode with 16,384 tokens
measured 33.72 rows/s, and throughput mode with 32,768 tokens measured 34.18
rows/s. Thus the largest apparent gain was only 0.35%, below run-to-run noise,
and none of the alternatives passed the predeclared `1e-4` score-parity gate.
The 16k and 32k conditions caused one and two median-score threshold flips.
Retain balanced scheduling; do not spend more A100 time tuning this token budget
for the frozen workload.

Repeated BF16 vLLM scores were not bit-stable even within a condition. The
baseline maximum per-row range was 0.0913 and its mean range was 0.00160, though
none of its 822 rows crossed the 0.5 threshold between repeats. This motivates
retaining repeat-level stability diagnostics alongside cross-condition parity.

## Merged-BF16 finding (2026-08-12)

Slurm job `30597577` compared dynamic LoRA with a safely merged BF16 model on
one A100. The same-job baseline measured 34.32 rows/s and 10.68k prompt tokens/s;
the merged model measured 39.61 rows/s and 12.33k prompt tokens/s, a 15.4%
steady-state speedup. Unlike dynamic LoRA, all three merged score arrays were
bit-identical.

Do not promote the merge yet: it failed the predeclared score-parity gate. Its
maximum absolute delta from the dynamic-LoRA median was 0.1036 and one of 822
predictions changed at the inclusive 0.5 threshold. That row was a label-zero
example (`dev-varied-deception-Qwen3.5-27B-None`, index 3247): dynamic LoRA
produced `[0.4688, 0.4688, 0.5]` across repeats while the merged model produced
an exact `0.5` tie every time. Pooled AUROC changed from 0.96688 to 0.96747 and
balanced accuracy from 0.91855 to 0.91736. This is promising but confounded by
the baseline's own numerical instability. Establish a deterministic reference
and an explicit equivalence rule before deciding whether the merge is safe;
do not proceed to FP8 under the current gate.

The Slurm job requests four CPUs and 64 GB rather than the repository default so
the vLLM frontend, engine process, and result serialization do not share one CPU
or approach host-memory pressure during a throughput measurement.

## Run

```bash
sbatch cluster/slurm/benchmark_classifier_throughput.sh
```

Override `CLASSIFIER_ADAPTER` to benchmark another vLLM-compatible adapter. Raw
results are written under `results/classifier_throughput/<job-id>/` and logs
under `logs/slurm/classifier_throughput/`; both remain ignored by Git.
