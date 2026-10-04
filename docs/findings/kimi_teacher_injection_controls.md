# Kimi K3 on injected honest APPS code: collaborator-reported diagnostic

Recorded 2026-10-04 from Michael Kenney’s writeup supplied by Jeremias Ferrao.
This is a summary of collaborator-reported results, not an independently audited
local evaluation. Reported upstream branch: `kimi-k3-teacher-controls`, commit
`0ad1778`, in `mkenney2/monitor-injection-evals`; artifacts under
`results/kimi_k3_teacher_controls/`. Raw predictions have not been collected or
checksum-verified in this checkout. No dataset rows or private correspondence
are reproduced here.

## Question and setup

Does the original student’s near-universal injection suspicion simply reproduce
the Kimi K3 teacher’s judgments? Michael rewrapped APPS trajectories with the
training teacher template, reportedly checking byte-identical student rendering
and the teacher template hash prefix `2418cc55`. The source snapshot is
`mjkenney/monitor-injection-evals@58e722bf`; the honest controls were rebuilt
with `build_honest_controls.py` from `feat/gleipnir-vllm-grid` and reportedly
matched the existing control manifest (hash prefix `71b337`). Full hash receipts
remain upstream and are not independently verified here.

The reported population contains 7,173 rows: 655 clean honest, 647 plain
backdoors, 1,941 injected backdoors, 1,965 injected honest, and 1,965
neutral-text honest. These are selected conditions, not the entire original
benchmark grid; matched ranking uses 647 paired task IDs per template. Code
comes from o3-mini and uses the three fixed transferred attack templates.

Model: `moonshotai/kimi-k3` through the project OpenRouter client, Makora only,
no provider fallback, temperature 0, reasoning effort none, top-5 logprobs,
max_tokens 64, prediction-line mode and prefix caching. Score is the normalized
binary probability from decision-token logprob 1 minus logprob 0. Hard verdict
is the generated `Prediction:` line and is distinct from a clean-calibrated
score alarm. The original training cache mixed Makora, Morph and Fireworks.

## Reported results

| Metric | GPT-4 template | GPT-4.1 template | GPT-4o template |
| --- | ---: | ---: | ---: |
| K3 injected-honest hard verdict 1 | ~0.2% | ~16% | ~73% |
| K3 matched injected-backdoor vs injected-honest AUROC | 0.971 | 0.949 | 0.918 |
| K3 mean injected-honest score | 0.040 | 0.320 | 0.581 |
| K3 mean neutral-text honest score | 0.278 | 0.131 | 0.169 |
| Fraction injected-honest score above neutral twin | 0.2% | 94.4% | 98.8% |
| K3 injected-honest alarm rate at clean ~2% FPR | 8.4% | 100% | 100% |
| K3 neutral-text honest alarm rate at same threshold | 99.7% | 78.5% | 90.8% |
| K3 injected-backdoor alarm rate at same threshold | 90.0% | 100% | 100% |

Reported plain-backdoor AUROC versus clean honest is 0.969. Clean honest hard
verdict 1 is ~0.2%; plain-backdoor hard verdict 1 is ~42%. The clean score
threshold is 0.0675, realizing ~1.98% FPR. Neutral text alone produces very high
alarms at that threshold, so it is a poor standalone diagnostic of injection
sensitivity in these OOD views.

For comparison, our original 4B matched injected AUROC is 0.643951, averaged
across templates (0.649131 / 0.609265 / 0.673455). Those student results are
historical, independently documented in our honest-control reports, and were
not rescored by Michael. AUROC measures ranking across the positive and negative
populations, not the fraction of matched task twins ranked correctly.

## Interpretation

The teacher is sensitive to some injections and its rubric includes attempts
to mislead a monitor. Code-backdoor labels and the teacher’s broader rubric can
therefore disagree on hostile comments appended to otherwise honest code.
Template-dependent teacher suspicion is compatible with a contribution to the
student’s behavior, but is not evidence of direct causal inheritance.

The teacher retains substantially better injected-class separation than the
original 4B student. The GPT-4 template also lowers teacher scores relative to
neutral text on ~99.8% of honest problems, whereas the student alarms on all
injected honest examples. These observations make uniform copying of the
teacher’s APPS response an inadequate explanation. They suggest amplification
or a different learned cue; they do not identify the training mechanism.

This scores new OOD APPS trajectories, not the actual training examples or
teacher targets responsible for learning. Provider differences, selected
conditions, reused tasks, fixed transfer templates and no local raw-data audit
limit comparisons. Treat transfer to our diagnosis as qualitative until exact
input/model/settings identities and raw summaries are checked. This was a
teacher-scoring diagnostic, not an intervention or retraining experiment, so
it cannot be described as a treatment that failed to help.

Michael proposed hard-label-adapter comparisons and format-only/plain-English
controls; these were not run in this reported experiment. No additional compute
is authorized or launched by this documentation update.
