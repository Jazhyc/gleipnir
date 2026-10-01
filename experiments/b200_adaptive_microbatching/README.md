# Adaptive B200 physical microbatches

Current recipe decision: after the completed execution audit, the user selected
compiled 16,384-token/max-8 adaptive batching for future B200 training. Use
`systems_screen@_global_: qwen35_4b_b200_adaptive` as the authoring profile.
The [decision record](../../docs/decisions/b200_adaptive_training_recipe.md)
distinguishes this explicit choice from the historical screen contracts below;
their failed parity gates remain intact. No new training run is launched.

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

Compiler-autocast audit: the pinned Torch 2.11 source defaults to
`backward_pass_autocast="same_as_forward"`, while the gradient canary uses BF16
forward autocast and backward outside autocast. The
[PyTorch semantics documentation](https://docs.pytorch.org/docs/2.11/user_guide/torch_compiler/torch.compiler_backward.html)
prescribes `"off"` for that pattern. After the unmodified compiled loss audit,
run a separately frozen `--mode compiled --backward-autocast off` canary with
the same eight inputs, loss, checkpoint policy, budget and maximum. Hypothesis:
matching the assumption reduces compiled gradient discrepancy without requiring
a forward-loss change. The wrapper scopes the override to the subprocess;
ordinary training settings and the original gate remain unchanged. Preserve
both results even if the hypothesis fails. Stop after the canary, with no
optimizer updates or recipe promotion.

The matched loss audit completed: compiled losses are 0.4411% below eager for
singleton accumulation and 0.3039% below for batch 8. Forward probabilities
differ by at most 0.014056 on these eight inputs. The within-compiled batching
gradient comparison still fails (relative L2 0.753615, cosine 0.659332).
The scoped backward-autocast override also fails (relative L2 0.647156, cosine
0.764318). Neither ran an optimizer update or established learning equivalence.
Retain both negative diagnostics, the original gate and fixed-batch recipe.
Fixed-physical-batch cross-backend gradients and actual AdamW updates remain
unmeasured; the findings distinguish those from these batching comparisons.

## Authorized gradient, actual-update and ten-step execution audit

The user explicitly requested all three remaining checks on 2026-10-01. This
new diagnostic contract authorizes optimizer updates even when gradient parity
fails. It preserves the failed timing campaign and its 0.05 gate, and does not
promote a recipe or authorize full training or held-out quality selection.

Hypothesis: close forward losses may coexist with gradient differences; actual
AdamW updates and short learning trajectories will establish whether the
differences materially change this bounded training workload. Compare four
conditions: eager/compiled crossed with singleton/adaptive physical batching.
Compiled execution uses the existing `same_as_forward` backward assumption;
the separately preserved `off` diagnostic did not meet parity. Model, targets,
rank, FP32 adapters, quantization, checkpoint policy and seed remain fixed.

`execution_audit.yaml` freezes ten logical updates of 32 traces from the existing
matched 320-row selection. Freeze one seed-0 permutation and sort only within
each logical batch. Singletons and 16,384-padded-token/max-8 adaptive partitions
see identical update membership. Trajectories retain untruncated materialized
inputs at the existing 29,696-token cap, Trainer AdamW groups/defaults, clipping,
linear scheduler and 3% warmup. In the ten-step schedule the first LR is zero;
record it rather than presenting that step as a nonzero update comparison.

The gradient/update probe uses the eight longest rows of this matched cohort,
tail-truncated to `[2048,2048,2048,2048,1024,512,256,128]`. These are explicitly a
new probe selection, not the historical global-longest-eight canary. Fixed
partition gradient comparisons isolate eager/compiled execution at singleton
and batch-8 shapes. Repeated same-backend gradients establish variability. The
actual probe update uses fresh Trainer AdamW state, clipping, and the configured
base LR 5e-5 without a scheduler, so its delta is nonzero. Compare all trainable
gradient and update elements, norms, cosine, relative L2, maxima and sign changes.

Restore and hash identical FP32 masters before every probe/trajectory; create
fresh optimizer/scheduler state. Measure before/after losses and each trajectory
step on the same eight training probes through one eager singleton eval path.
Report every step's training loss, probe losses, LR, gradient norm, timing,
partition and example membership, plus cumulative parameter-update comparisons.
Save diagnostic final FP32 masters on the volume. No serving artifact is exported.

Stop on OOM, nonfinite tensors, input/hash/restoration drift or runtime error;
retain failures and completed measurements. Ten steps are diagnostic evidence,
not final-quality equivalence. No throughput gain or epoch ETA is promoted from
these runs. All conditions run even if the gradient comparison misses the old
gate, because measuring subsequent updates is now explicitly authorized.
Inspect startup every 30–60 seconds and actual step records thereafter. No
after-turn scheduling mechanism is available; active-turn checks cover this
campaign and no later wakeup is promised. Leave the existing $6.79/hour Pod
running afterward.

```bash
GLEIPNIR_COMMIT=COMMITTED_REVISION .venv/bin/python \
  experiments/b200_adaptive_microbatching/run_execution_audit.py \
  --config experiments/b200_adaptive_microbatching/execution_audit.yaml
```

The execution audit completed all four probes and forty trajectory steps without
OOM or unexpected compiler fallback. Fixed-partition eager/compiled gradients
differ by 10–11% relative L2 and actual AdamW updates by 32–34%, while final
common probe losses span 0.927562–0.940055 (1.35%). Adaptive policies realized
physical sizes 1, 2, 4 and 8. Step-compute timings are promising but exclude
evaluation/setup/reset/export and share warm caches; no full-loop gain or ETA
is promoted. These cohort-specific probes differ from the historical stress
inputs. See the [complete findings](../../docs/findings/b200_execution_audit.md)
for curves, individual-example limits, provenance and retained checkpoints.
The original parity gate and fixed-batch selection remain unchanged.
