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

Complete the predeclared four-condition sweep once. Do not tune the token budget
against individual datasets or quality metrics. Reject a condition on OOM,
missing scores, prompt-order mismatch, or parity failure, and report any
within-condition threshold instability. Select the fastest passing condition by
median rows/s. Only then benchmark a merged BF16 adapter,
FP8 with a new parity gate, and a packed Transformers final-position forward in
that order. A100 results select configurations to retest on H100; they are not
an H100 throughput estimate.

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
