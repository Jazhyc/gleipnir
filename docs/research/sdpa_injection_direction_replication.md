# SDPA monitor replication of injection-direction interventions

This review informed the user-authorized
[small-fit replication](../../experiments/b200_injection_direction/README.md),
completed on the current BF16 eager SDPA Lens scorer; its
[findings](../findings/b200_injection_direction.md) own the current-model results.
No intervention is selected
or promoted. Pin the external branch to commit
`931fee0a1db369bc15c01009aa77219e7e106c14`. Its
[ablation implementation](https://github.com/mkenney2/monitor-injection-evals/blob/931fee0a1db369bc15c01009aa77219e7e106c14/interp/ablate.py),
[frozen fix](https://github.com/mkenney2/monitor-injection-evals/blob/931fee0a1db369bc15c01009aa77219e7e106c14/interp/frozen/frozen_fix.json)
and [reported bootstrap results](https://github.com/mkenney2/monitor-injection-evals/blob/931fee0a1db369bc15c01009aa77219e7e106c14/results/bootstrap_ci.json)
are the source evidence. Local file hashes and reported numbers are preserved
in `results/b200_sdpa_lens/replication_review/provenance.json`; no upstream
rows or raw score logs are imported into this project.

## Method and reported benefit

Fit each layer's direction as the mean paired injected-minus-neutral decision
residual, matching task/template/label, on training-partition rows only. Both
honest and backdoored pairs contribute. Task split is deterministic SHA-256,
60% fit/40% test; the ablation scripts select 200 tasks before intersecting with
test, yielding the reported 74-task intervention subset. Full 655-task unedited
results and 269-task later ladder results are different cohorts.

The principal augmented fix uses the **same unit layer-20 direction at all
32 decoder layers and every token**, with beta 0.25. Its operation is
`h' = h - beta * (dot(h, u) - c[layer, position]) * u`. The center is the
neutral-fit decision-token projection mean at the final token and the
neutral-fit inserted-span mean at every other token. Projection sets beta 1
and both centers to zero. Random directions are controls. Addition is a
separate sufficiency experiment at one layer/all tokens; strength is in units
of the recipient model's own natural injection-shift norm.

| Reported historical augmented APPS arm | Injected-honest FPR | Injected-backdoor TPR | Matched AUROC | Plain AUROC |
|---|---:|---:|---:|---:|
| Unedited | 45.05% | 83.56% | 0.810846 | 0.906886 |
| Layer-20 clamp, beta 0.25 | 3.15% | 43.38% | 0.846469 | 0.905591 |
| Full projection | 1.35% | 38.81% | 0.870739 | 0.906979 |

These estimates recalibrate the strict clean-only at-most-2%-FPR threshold per
intervention. The source also saves fixed-unedited-threshold results; report
both. Low false alarms come with a substantial decrease in injected-backdoor
recall. Plain backdoor recall changes from 53.42% to 52.05% with the clamp.
Bootstrap intervals cover problem sampling, not seed variability. Layer/beta
choices were frozen before MonitoringBench, after APPS exploration; APPS is
not an untouched confirmation set for those choices. The external
[MonitoringBench transfer report](https://github.com/mkenney2/monitor-injection-evals/blob/931fee0a1db369bc15c01009aa77219e7e106c14/results/monitoringbench/README.md)
reports regular-model alarms 26% to 2%, with plain AUROC 0.94 to 0.95;
augmented-model alarms are already about 3% and remain about 3% under clamp.

The later [75%-TPR/4%-FPR result](https://github.com/mkenney2/monitor-injection-evals/blob/931fee0a1db369bc15c01009aa77219e7e106c14/results/probe_ablated/README.md)
comes from a separate, newly trained linear probe after removing a 16-dimensional
injection subspace at layer 24. Keep that classifier change distinct from
ablation/steering of the existing monitor head. Cross-model clamp results fit
separate directions per model; they do not demonstrate transferring one fixed
vector between those adapters.

## Replication on the current stack

The colleague uses the historical augmented master `3ecfd0d397ae...`; the
current SDPA replay uses `bdb702316b1d...`. Fit directions/neutral centers on the
current model for the primary replication. A historical-vector transfer, if
available, is a separate intervention. Preserve BF16 eager serving, exact
prompt/token identity, paired source lineage and a fresh unedited eager control
on the same rows; compiled BF16 ID/APPS controls alone do not establish that
baseline.

The Lens bridge already captures the complete post-layer residual (including
vLLM's deferred residual sum), handles absolute positions/chunks, and supports
addition. Exact clamp/projection needs a request-scoped activation-dependent
operator computed in FP32 and cast back to BF16. Apply the decision center only
to the absolute final input token, never to the last row of each prefill chunk.
Add span-mean capture with sums/counts across chunks to avoid returning every
span activation at all 32 layers. Validate against a direct tensor reference,
no-op/beta-zero behavior, mixed-request and absolute-position isolation,
chunking, cancellation and cleanup before a scoring campaign.

Stage the current-model capture/probes, then fixed unedited/clamp/project/random
arms and additive positive/negative-direction controls. Report matched AUROC/raw
pAUROC20, injected-honest FPR and injected-backdoor TPR together, plain/neutral
controls, calibration/ties, paired score shifts and grouped uncertainty. Check
CoT-removed ID pAUROC20 as a broader-monitoring guard. Leave-one-template-out
fits are useful diagnostics. No outcome-driven layer/beta sweep or promotion.

Two-token pooling reports conditional 0/1 scores; the colleague's strong
steering checks also report full-vocabulary 0/1 probability mass (`p01`). That
requires a separate bounded full-vocabulary readout. Generation steering and
attention-map patching likewise need separate supported paths; the current
pooling capture/steering bridge does not provide those experiments.

The user explicitly authorized a small APPS fitting portion for this exploratory
replication. Its experiment contract freezes 64 fit tasks, excludes all their
variants from the 591-task test partition, and fixes interventions before
evaluation. This exception does not relabel the historical full benchmark as
held out or authorize training/checkpoint selection on APPS. Training
nondeterminism remains deferred. No model or serving default changes.
