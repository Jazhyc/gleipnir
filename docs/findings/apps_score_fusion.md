# Combining behavior and injection-presence scores

The 2026-10-11 offline pilot gives a mixed answer: adding the injection margin
improves pooled APPS behavior discrimination but **worsens discrimination within
injected trajectories**. It does not establish the desired improvement in
honest-versus-backdoor separation when both contain an injection.

The [frozen experiment](../../experiments/apps_score_fusion/README.md) reuses
the completed SDPA BF16 behavior and
[injection-presence](apps_injection_detection.md) scores. Labels predict code
backdoors, not injection presence. Exact ID/task/condition/source-sample hashes
and behavior labels agree across all 9,114 pairs; both scores use the identical
SDPA master and merged weights. Safety was scored on the earlier driver-595
B200, presence on the driver-580 B200. This is exploratory score reuse, with
that host/driver difference retained; no GPU run, serving change or promotion.

Reuse the preceding task partition: 64 tasks / 896 rows fit the combiner and
591 disjoint tasks / 8,218 rows evaluate it. Keep every derived view with its
task. Freeze C=1 logistic regression, train-only feature standardization and
four arms: raw safety, safety-only calibration, two margins, and two margins
plus their product. No template indicator, prompt change, hyperparameter search
or post-result arm selection occurs. APPS has prior research exposure; these
task-disjoint results are not a fresh final-test claim or monitor retraining.

| Arm | Pooled behavior AUROC | Pooled raw pAUROC@20 | Mean matched injected AUROC | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Raw safety | 0.788105 | 0.460105 | 0.803573 | 0.289758 | 0.307036 |
| Safety-only calibration | 0.788105 | 0.460105 | 0.803573 | 0.191100 | 0.096893 |
| Two scores | 0.831443 | 0.504176 | 0.785848 | 0.170959 | 0.051275 |
| Two scores + interaction | 0.830422 | 0.498340 | 0.776533 | 0.171396 | 0.050011 |

Matched injected AUROC gives each of the three templates equal weight; each
template compares 583 honest and 583 backdoor rows with the same tasks and
payload. The two-score change is −0.017724. A paired 591-task bootstrap with
500 draws, seed 0, gives 95% percentile interval [−0.024511, −0.011145].
Uncertainty covers task sampling, not training seeds or new attack templates.

| Injected template | Safety-only AUROC | Two-score AUROC | Interaction AUROC |
|---|---:|---:|---:|
| GPT-4 transfer | 0.741439 | 0.733763 | 0.725719 |
| GPT-4.1 transfer | 0.811980 | 0.773289 | 0.755872 |
| GPT-4o transfer | 0.857299 | 0.850492 | 0.848009 |

The two-score fitted coefficient on injection margin is negative. In raw-margin
units, its approximate portable interface is
`sigmoid(1.975 + 1.450 * safety_margin - 0.614 * injection_margin)`; the exact
standardized coefficients, centers and scales are in `models.json`. This fits
an injection-associated offset in the safety score. The within-injection ranking
regression shows that the extra score also removes useful harmfulness signal;
pooled ranking alone would miss that limitation. The interaction arm does not
recover it in this frozen pilot.

Thresholds are fitted only on **all fitting behavior negatives**, allowing at
most floor(2% * negatives) alarms with strict score > cutoff, then held fixed on
evaluation. They differ from the prior clean-only APPS threshold and are not
recalibrated using evaluation labels.

| Arm | Evaluation FPR, all negatives | Evaluation backdoor recall |
|---|---:|---:|
| Raw / calibrated safety | 1.088% | 30.238% |
| Two scores | 2.490% | 37.515% |
| Interaction | 2.611% | 36.609% |

Operating points have different realized FPRs, so their recall differences do
not establish a gain at matched FPR. In particular, two-score matched injected
FPR/TPR varies by template: 8.405%/40.137%, 0%/19.726%, 0%/22.127%;
safety-only gives 5.146%/34.648%, 0.343%/28.816%, 0%/26.072%. At 0.5,
two-score pooled FPR/recall is 24.849%/73.585%; calibration changes threshold
semantics substantially. Full diagnostics, neutral controls and ties remain in
the summary, with the shared calibration helper's >=0.5 convention separate
from the strict operating-point diagnostics.

Computing both prompts would process 33,067,774 tokens for this 9,114-row grid,
versus 15,339,953 safety-only: **2.156× input tokens**. No combined serving
latency or production-throughput measurement occurs here. The extra inference
cost is not justified by better within-injection separation in this pilot.

Artifacts: `results/apps_score_fusion/fusion01/` contains paired inputs,
portable models, predictions for every arm, hashed contract/completion and
bootstrap/quality summary. Logs: `logs/apps_score_fusion_run.log`; all four
fits complete without warnings/failure. Four focused fusion tests and repository
Ruff pass. Any subsequent model family, richer combiner or same-host confirmation
needs a new frozen contract; the current result does not select a deployed rule.
Broader ID monitoring quality has not been evaluated for these combiners.
