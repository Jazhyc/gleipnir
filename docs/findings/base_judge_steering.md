# Augmented-vector steering in base-model JudgeDeceiver judging

The original APPS vector has a **large causal effect in base-model judging**,
dominated by choosing answer B. At +2x, B is chosen in **99.93%** of rows and
pooled accuracy falls from **78.44% to 50.07%**. The same base/vector still
raises clean/neutral APPS alarms to 100%. This is broad steering influence
with task/readout-dependent behavior, without improved injection judging.

The user requested a domain-transfer test of the original augmented-monitor
injection vector on the **unadapted Qwen3.5-4B base**, rather than another
projection on a monitor adapter. The
[experiment contract](../../experiments/base_judge_steering/README.md) fixes
the model, vector, doses, cohorts, readouts and stop conditions before outcomes.
No direction or classifier is fitted on JudgeDeceiver and no model is promoted.

## Matched domain/readout comparison

Use base revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` with verified
configuration/weight-shard checksums and no adapter loaded. The original unit
layer-20 axis and natural mean shift come from the unfiltered augmented SDPA
monitor, archive SHA256
`8c3c6e50ee1d3880d854754f9899f5684ae2e2c686479c3ceaaf3677d9eecfa7`.
Add the natural shift at **layer 20 only, every token**, +1x/+2x, with no
projection or norm matching. Vector norms are **7.072298 / 14.144595**; these
are the donor augmented-model shift sizes, not recipient-normalized doses.
Keep the original seed-0 random axis at equal norms as a perturbation control.

Five JudgeDeceiver arms share the same BF16 eager Lens engine and original
**4,188 A/B prompts**: 504 clean, 1,842 preferred-injected, 1,842 disfavored-
injected, 252 pairs, both orders and **six held-out query groups**. Native
tokens total 1,521,530 per arm. A/B preference labels and the .5 cutoff stay
fixed, with ties choosing B. Source revision, construction labels, dependent
variants, grouping and unresolved upstream license follow the
[original dataset contract](../../experiments/judge_injection_continuation/README.md).
Preferences are separate from harmfulness and false-positive rates.

A matched base-model APPS control uses the existing frozen 128-task subset:
**512 clean/neutral honest rows**, excluding all 64 fitting tasks, injected
views and backdoors. All five arms are freshly scored on the same base/vector.
This prevents changing CAFT to base from being mistaken for a domain effect.
Its one unedited clean-only 2%-FPR threshold stays fixed; 128 clean rows give
limited operating-point resolution. APPS and judging use different prompts,
tasks and readouts, so this comparison cannot isolate semantic domain from
prompt or answer-token effects.

## Results

| Base judging arm | Clean accuracy | Preferred-injected accuracy | Disfavored-injected accuracy | Pooled accuracy | Pooled AUROC | Raw pAUROC20 | B choices |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unedited | 98.41% | 52.71% | 98.70% | 78.44% | 0.90534 | 0.66915 | 29.97% |
| Augmented vector +1x | 84.13% | 49.02% | 85.50% | 69.29% | 0.86068 | 0.62496 | 74.98% |
| Augmented vector +2x | 50.20% | 50.00% | 50.11% | 50.07% | 0.85240 | 0.63849 | 99.93% |
| Equal-norm random +1x | 98.02% | 53.20% | 98.75% | 78.63% | 0.89762 | 0.66737 | 30.59% |
| Equal-norm random +2x | 96.83% | 52.39% | 98.26% | 77.91% | 0.90165 | 0.69586 | 27.91% |

The dataset has equal counts of correct-A/correct-B preferences. At +2x,
correct-A accuracy is **0.14%**, correct-B accuracy **100%**. For clean rows,
the corresponding rates are **0.40% / 100%**. Preferred-injected rows choose B
universally, yielding **0% / 100%** accuracy by correct label. This also changes
the already asymmetric unedited preferred-injected behavior: correct-A/B
accuracy was 96.53%/8.90%. Large order-dependent flips therefore do not imply
better rejection or recognition of injection content.

Positive +1x/+2x shifts the pooled raw B-minus-A margin by **+1.247 / +1.529**,
while the label-adjusted correct-answer margin decreases **-1.182 / -1.450**.
There are **1,885 / 2,930 decision flips**, versus 54/86 for random controls.
This is not one constant offset: clean correct-A margins move toward B by
3.366/4.080, while clean correct-B margins decrease by 1.227/1.588 and mostly
retain B. The effect reduces discrimination and changes the decision criterion.
Ranking remains substantial at +2x (AUROC 0.8524), despite almost universal B
choices at the fixed .5 threshold; preference information is not wholly erased.
Unique scores compress from **96** unedited to **38/24** at +1x/+2x; exact-half
ties increase from **47** to **342/100**. At +2x, p(B) strictly exceeds .5 in
**97.54%** of rows, so the strong B bias persists without relying on ties.

Source-/query-macro accuracy changes from **76.03%/76.94%** unedited to
56.41%/61.04% (+1x), 50.04%/50.05% (+2x). All four sources are near 50% at +2x.
Pooled Brier/ECE changes from **0.14482/0.12470** to 0.17753/0.17335 and
0.23141/0.25818. Mean p(correct) falls from 0.74853 to 0.62376/0.57219.
Full source/query, paired clean/injection, calibration and tie diagnostics
remain in the summary. Six dependent query groups bound generalization;
there is no independent-rows significance claim.

A/B answer mass stays high: mean/minimum is **0.99878/0.97210** unedited,
**0.99139/0.96813** (+1x), **0.98054/0.94658** (+2x); no row falls below .5.
Thus the judging collapse is a decision bias within the requested A/B surface,
not failure to put probability mass on A/B. Auxiliary unrequested 0/1 mean
mass is only 1.58e-6 unedited and 7.37e-6/5.13e-5 steered. Its mean 1-minus-0
margin shifts **-0.985/-0.530**, unlike the positive APPS shifts below. These
conditional unrequested-head values are diagnostics, not harmfulness scores
for the judging task or evidence that the model literally answers 0/1.

| Same-base APPS honest control | Clean FPR | Mean neutral FPR | Pooled FPR | Mean paired 1-minus-0 margin shift |
|---|---:|---:|---:|---:|
| Unedited | 1.56% | 2.08% | 1.95% | 0 |
| Augmented vector +1x | 76.56% | 90.89% | 87.30% | +2.1665 |
| Augmented vector +2x | 100.00% | 100.00% | 100.00% | +2.4343 |
| Equal-norm random +1x | 1.56% | 2.34% | 2.15% | -0.0769 |
| Equal-norm random +2x | 1.56% | 2.86% | 2.54% | +0.1963 |

APPS uses its fixed base clean-calibrated **0.5** threshold with strict `score >
threshold`, unlike judging's B-on-tie convention. Ties matter: the unedited
APPS pooled FPR is 3.52% under `score >=0.5`, versus 1.95% under the frozen
strict rule. At +1x those rates are 96.88%/87.30%; at +2x both are 100%.
The positive-control mean monitoring scores rise from 0.15960 to 0.56225/0.62735,
with answer-mass minima 0.99820/0.98567 and means 0.99956/0.99776.

The vector affects **both tasks**, raising monitoring scores on APPS and causing
strong positional/answer bias on JudgeDeceiver. This disfavors an exclusively
monitoring-context causal role, while preserving the distinction between
steerability and useful semantic transfer. A broader verdict/readout bias is a
candidate explanation, not an established single-feature interpretation.
Prompt and decision-token differences, nonlinear propagation, one donor/seed,
two doses and one random direction limit that interpretation. No prompt remap,
additional fitting or intervention selection follows these outcomes.

## Validation and execution

The verified CAFT Lens process is retired sequentially, preserving its
weights/merges, scores and persistent caches. The unadapted base is loaded by
changing only model identity in the passing BF16 eager command. Dynamic LoRA
is forbidden; no training, new merge, dependencies or capacity lifecycle occurs.

The fresh base 0/1 canary passes with MAE **0.0073178**, correlation
**0.9992609** against the archived base reference. Native BF16 dispatch,
Lens/client/isolation/operator checks pass. The separate 30-row A/B head
check passes MAE **0.0063638**, correlation **0.9996582**, using independent
FP32 selected embedding rows on the same normalized hidden. This is a head
arithmetic check, not whole-backbone FP32-master A/B parity. Zero addition and
next-plain requests restore A/B logits exactly; a nonzero addition has effect.

The judging scorer adds opt-in preservation of unrequested 0/1 mass/margins
alongside the requested A/B readout; its historical default remains unchanged.
Thirteen focused tests pass, covering label/prompt binding, A/B bias versus
correct-answer margins, monitor answer-mass validation and replacement command
identity. Scoped Ruff/diff checks pass. The inherited OMP_NUM_THREADS=4 warning
flags possible CPU contention; the matched setting remains unchanged.

Artifacts use `results/base_judge_steering/base01/` and logs
`logs/runpod/base_judge_steering/`. Original sources/configuration, canonical
workloads, base/readout/native gates, all batch scores and full readout
diagnostics remain persistent. Weights stay remote and caches remain intact.

All **23,500 primary scores** (20,940 JudgeDeceiver + 2,560 APPS controls) pass
the independent audit: unadapted model identity, input/source/score hashes,
canonical prompt/label/token/order bindings, finite A/B and 0/1 logits/margins,
all **185 primary batches** and recomputed five-arm metrics. Canaries and smoke
checks are separate diagnostic requests. Artifacts and receipt copies are
collected locally without model weights. The passing base Lens scorer remains
warm on port 8010, recorded in `results/base_judge_steering/base01/server.json`;
CAFT is not automatically restored and capacity remains running.

JudgeDeceiver's five arms take **66.81 / 70.05 / 69.89 / 69.44 / 70.50s**
for 1,521,530 input tokens each, or **22,774 / 21,722 / 21,771 / 21,910 /
21,582 input tokens/s** (plain, positive +1x/+2x, random +1x/+2x). APPS control
times are **11.18 / 11.36 / 11.04 / 11.07 / 12.32s** for 760,060 input tokens
each, or **67,977 / 66,933 / 68,856 / 68,640 / 61,679 input tokens/s**.
These are single-pass end-to-end costs, including full readouts; the different
lengths, readouts and concurrency preclude interpreting them as a systems
throughput comparison. Per-request queued timings remain in raw score files.
