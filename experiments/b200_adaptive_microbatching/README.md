# Adaptive B200 physical microbatches

Hypothesis: keeping long traces as singletons while batching short traces reduces
training time relative to the validated half-checkpoint batch-1 recipe. The
intervention changes physical batching within each optimizer update, preserving
the same 32 examples per update, teacher targets, rank, seed, objective, and LR.
No full training campaign or ID/OOD quality selection is authorized or performed.

Reuse the preceding B200 screen's fixed stratified 320-row mixed-training cohort
and longest-32 preflight. Trainer sees logical batches of 32 with accumulation 1.
Inside `training_step`, sort only those 32 features by actual materialized input
length, then split them into power-of-two microbatches. Each microbatch mean is
weighted by its example count divided by the actual logical-batch size. Clip,
step AdamW, and advance the scheduler only after the complete logical batch.
Partial final batches use their actual example count. Prompts remain independent
sequences; this is not concatenation or sequence packing.

The matched singleton control uses the same logical-batch path and within-update
ordering. Candidates have padded-token budgets of 8,192 (maximum physical batch
4) and 16,384 (maximum batch 8). A sequence longer than its budget is permitted
only as a singleton, retaining the 29,696-token context cap. These budgets are
bounded trial policies, not analytical memory guarantees. All use SDPA, twelve
linear checkpoints, rank-128 NF4 QLoRA/BF16 compute, FP32 adapters, selected-position
logits, FLA 0.5.2, and the existing selective compilation policy.

Before timing, require the existing compile canary and longest-32 training update,
plus a same-weights gradient canary comparing singleton and adaptive partitions
over eight unequal-length inputs up to 2,048 tokens. Compare every trainable
gradient, require a nonzero reference norm and global relative L2 error <=0.05,
clear gradients afterward, and perform no optimizer update. This also exercises
batch 8 at the aggressive budget. The representative screen profiles the
intermediate lengths; stop on OOM, nonfinite output, checksum drift, incomplete
coverage, failed parity, or more than 24 Dynamo graphs. Preserve failures and
freeze any revised policy as a separate campaign; do not silently retry smaller
batches.

The first pass records synchronized per-microbatch elapsed time and peak memory,
alongside complete optimizer-update timings, graph counts, padding, realized
microbatch sizes, and logical coverage. Memory-stat resets preserve an aggregate
high-water mark for the campaign. Profiling synchronization applies to every
condition; promising candidates require a complete cached repeat with profiling
disabled before recommendation. Compare full loops, not isolated short updates.
Require at least 5% throughput gain over the matched control. The two budgets
are selected only by systems measurements; no validation/test scores are used.

Initial support is single-device, dropout-free per-example binary hard/soft
losses with proportional random sampling and no in-training evaluation. Auxiliary
completion, pairwise, MIL, prefix, ordinal, dataset-reweighted and distributed
objectives fail closed. Existing fixed-microbatch training retains its behavior.

The user authorized empirical iteration on existing B200 Pod `alzfug70g5237b`
at $6.79/hour, leaving it running afterward. Reuse the persistent compatible
cache; do not launch additional capacity. Keep FP32 masters on the volume and
collect contracts, metadata, summaries, and logs locally. Source revisions and
cache provenance accompany each run. Active-turn startup checks occur every
30–60 seconds; this session has no verified after-turn agent scheduler.

```bash
.venv/bin/python -m gleipnir.monitoring_systems_screen prepare \
  --config experiments/b200_adaptive_microbatching/config.yaml
.venv/bin/python -m gleipnir.monitoring_systems_screen run \
  --config results/b200_adaptive_microbatching/resolved_config.json
```

Prepare on the Pod and use `launch.sh` with the synced commit in `GLEIPNIR_COMMIT`.

The first GPU preflight stopped before any optimizer update: batch 8 over unequal
lengths produced relative gradient L2 error 0.51951 (threshold 0.05). Preserve
that failed screen. The next bounded diagnostic uses `diagnose.py --mode eager`
and the same frozen preflight job. It runs the eight-input canary only, recording
decision logits, mean losses, gradient norms/cosine and the largest parameter
differences. `--mode compiled` provides the matched compiled diagnostic. Neither
diagnostic trains or selects a new recipe; failure keeps the parity threshold.

The eager diagnostic also fails (relative L2 0.13763, cosine 0.99050); its mean
loss changes by 0.157%. The next diagnostic adds `--precision-mask-probes` to
isolate FP32 LM-head projection, omission of causal right-padding masks, and
their combination, plus conservative physical maxima 4 and 2. These are bounded
same-weight canaries only. Restore model
methods after each probe, retain the original failure and report all variants.

The batching probes are complete: maxima 2, 4 and 8 all miss the 0.05
gradient-parity gate; the best mask/precision probe is 0.09221. No timed training
condition ran, and no adaptive policy is promoted. Keep the validated batch-1
recipe. Implementation remains opt-in/experimental. See the
[recorded findings](../../docs/findings/b200_adaptive_microbatching.md) for every
probe, artifacts and limits on this conclusion.

Kernel audit: run `diagnose.py --mode eager --maximum-microbatch-size 1` on the
same eight inputs to establish the identical-shape repeatability noise floor.
Both reference and actual gradients then use the same singleton partition and
order. Hypothesis: repeatability error is much smaller than cross-shape error;
the baseline remains unchanged regardless. Retain the 0.05 gate, no optimizer
updates, original frozen job, and a separate output directory. This can rule out
large nondeterminism for these inputs, but cannot establish cross-shape kernel
correctness or compiled-path repeatability.

The singleton repeatability audit passed: relative gradient L2 0.005119,
cosine 0.999987, and exactly identical mean losses. This is much smaller than
the cross-shape discrepancy, but does not identify its cause or validate an
adaptive policy. The installed FLA includes the known Blackwell forward-state
guard; other upstream race reports concern different hardware/load and were
not reproduced here. Keep the original recipe and parity threshold.

Matched loss audit: run `diagnose.py --mode compiled` with the original maximum
8 and a new output directory to collect the enhanced loss/logit diagnostics
missing from the initial compiled failure. Compare both singleton and batch-8
mean losses and per-example decision logits against the recorded eager run,
with unchanged frozen inputs, seed, weights and teacher targets. Hypothesis:
forward-loss agreement can be much closer than gradient agreement. No optimizer
update occurs; preserve the existing 0.05 gradient gate and baseline selection.
A small mean-loss gap cannot alone establish equivalent subsequent updates or
learning trajectories. Stop after the eight-input canary or any nonfinite/OOM
failure; this audit does not authorize a full training campaign.
