# Add the augmented-model vector to CAFT honest examples

User-requested causal sensitivity check: add the original augmented-model
injection vector to the CAFT checkpoint on honest APPS examples. Hypothesis:
the original direction still increases suspiciousness away from the already
saturated injected-honest population. This tests steering sensitivity, not
whether CAFT caused rerouting or whether the direction is necessary.

Reuse the same warm BF16 eager Lens engine, CAFT merge/master and passing native/
functional/operator receipts. Verify process/source/runtime identities and a
fresh 20-row no-op against cached eager scores (MAE <=0.005, correlation >=0.995).
No training, fitting, restart, merge, new capacity or dependency change.

Select all **591 clean + 1,773 neutral honest rows = 2,364 rows** from the same
591-task APPS test partition. Exclude every direction-fitting task, injected
honest view and backdoored example. Keep all three neutral templates. Reuse
cached same-engine unedited scores; do not rescore the baseline. Retain the
original 128-task answer-mass subset (512 honest rows per arm).

Freeze four added-vector arms: original augmented mean shift at **+1x / +2x**
and the original seed-0 random unit direction at equal norms. Add only after
**layer 20**, at every token, with norm matching disabled and no projection.
One unit of strength is the original augmented shift norm **7.072298**, not
the CAFT model's shift norm. Thus vector norms are 7.072298 and 14.144595.
The two predetermined doses follow the earlier additive comparison; no dose,
layer, template, random-seed or checkpoint selection after outcomes.

Use c64/batches128. Report each condition's fixed unedited clean-calibrated
threshold FPR, FPR at 0.5, continuous scores and 1-minus-0 logit margins, paired
shifts, calibration/ties and answer mass. Never recalibrate the threshold after
adding the vector; that would obscure this causal alarm test. All labels are
harmless, so AUROC/pAUROC are undefined. No promotion or requested ID evaluation.

Stop on identity/input/source/runtime drift, failed no-op, invalid vector,
nonfinite outputs, wrong labels/membership, missing coverage, cleanup failure
or OOM. Preserve partial batches and failures, keep weights remote and the
passing engine warm; no capacity lifecycle. Monitoring is active-turn only.
Inputs and geometry are bound in `config.json`; results use
`results/caft_honest_steering/honest01/` and logs
`logs/runpod/caft_honest_steering/`. The runner reuses the resumable Lens scorer.
