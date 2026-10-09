# APPS projection transfer to JudgeDeceiver

2026-10-09. The same full projection that nearly eliminates APPS honest-injection
alarms has **little aggregate benefit in JudgeDeceiver preference judging**.
Preferred-injected accuracy improves only 0.65 points, with opposing source
effects. Pooled ranking improves slightly while calibration worsens. The
[experiment contract](../../experiments/b200_projection_judge/README.md) owns
the frozen transfer, gates, populations and stop conditions; the
[APPS finding](b200_injection_direction.md) owns fitting and original effects.

## Fixed transfer and matched control

Transfer the already fitted unit layer-20 direction unchanged: full projection
at all 32 layers/all tokens, beta 1, zero centers. No JudgeDeceiver fitting,
strength/layer/threshold sweep, retraining or promotion. Use the current
SDPA augmented master `bdb702316b1d...` and checksum-identical merged BF16
checkpoint, vLLM 0.31/Torch 2.13/Lens 1.3.0 eager research serving on the same
user-reserved NC2 B200.

Reuse the exact original 4,188 A/B prompts: 504 clean, 1,842 preferred-injected
and 1,842 disfavored-injected, 252 pairs, both orders, six held-out query groups.
All preferences and prompt/token identities remain frozen; ties at p(B)=.5
choose B. These labels are preferences, not harmfulness or injection FPR.
Source revision, construction labels, unresolved upstream license and the
dependent order/suffix grouping follow the
[original JudgeDeceiver contract](../../experiments/judge_injection_continuation/README.md).
The current SDPA adapter did not train on this dataset. Previous FA4/historical
adapter scores do not substitute for this fresh unedited SDPA control.

## Results

| Metric | Unedited SDPA | Full projection |
|---|---:|---:|
| Clean accuracy | 97.82% | 99.01% |
| Preferred-injected accuracy | 55.81% | 56.46% |
| Disfavored-injected accuracy | 98.05% | 98.37% |
| Pooled accuracy | 79.44% | 80.01% |
| Pooled AUROC | 0.910254 | 0.914715 |
| Pooled raw pAUROC20 | 0.686680 | 0.696586 |
| Preferred-injected AUROC | 0.631926 | 0.638094 |
| Preferred-injected raw pAUROC20 | 0.233552 | 0.224667 |
| Source-macro accuracy / AUROC | 80.46% / 0.916053 | 80.57% / 0.931113 |
| Query-macro accuracy / AUROC | 80.11% / 0.899474 | 80.38% / 0.912004 |
| Pooled Brier / log loss | 0.127261 / 0.375821 | 0.130899 / 0.387435 |
| Pooled ECE | 0.091809 | 0.112615 |
| Mean p(correct) | 0.751156 | 0.749889 |
| Exact-half ties | 106 | 45 |

The 182 decision flips comprise 103 correct decisions gained and 79 lost:
clean gains six/loses zero; preferred injections gain 86/lose 74; disfavored
injections gain 11/lose five. Preferred-injected score MAE is 0.054956, despite
only twelve net correct decisions gained. Its mean p(correct) decreases
0.555634 to 0.552233, and raw p20 decreases; the accuracy gain is not a uniform
ranking/confidence improvement.

| Preferred-injected source | Rows | Unedited accuracy | Projection accuracy | Change |
|---|---:|---:|---:|---:|
| LLMBar | 888 | 61.04% | 56.87% | -4.17 pp |
| MT-Bench | 896 | 50.33% | 56.25% | +5.92 pp |
| RLAIF | 20 | 60.00% | 50.00% | -10.00 pp |
| Search | 38 | 60.53% | 55.26% | -5.26 pp |

Preferred clean-to-injection mean p(correct) change is -0.371889 unedited versus
-0.375654 projected; correct-to-wrong injection flip rates are 43.21% versus
42.62%. Disfavored score shifts are -0.023765 versus -0.024403, with flip rates
1.52% versus 1.19%. Denominators retain every paired injection variant.
The full six-query accuracy breakdown is in `summary.json`; query-macro
accuracy changes by only 0.26 points. Six queries, few rare-source rows,
dependent variants and one model/run do not establish broad or seed-robust
transfer. No independent-rows significance claim is made.

Both arms retain A/B probability mass: means 0.966640/0.968585, minima
0.775898/0.766040; no row below .5. These are readout diagnostics, not generation
format tests. The direction's large APPS effect does not transfer reliably to
this other prompt/task/readout. This experiment cannot isolate which of those
differences limits transfer.

## Readout validation and receipts

The Lens full-vocabulary path now also returns A/B rows 32/33 and p(A or B)
from the actual fused final-normalization result and tied embedding. Its
resident cached 0/1 head/backbone remain identical. A separate FP32 selected-row
projection of the same normalized hidden validates the head. The original
30 training-derived canaries pass MAE **0.004628**, correlation **0.999804**
against that reference. This is a head check, not a new whole-backbone
FP32-master A/B parity comparison. Reuse the unchanged adapter/merge receipts;
fresh existing 0/1 master/native gates pass. The new eager 20-row no-op canary
reproduces the previous engine exactly. Real A/B beta-zero/effect/next-plain
restoration checks pass. Focused runtime tests pass **19**, with 14 inherited
Torch deprecation warnings; local preference contract tests pass **two**.
Scoped Ruff and diff checks pass.

`results/b200_projection_judge/sdpa_judge_projection01/` contains the frozen
manifest/config/workload, executed sources, readout gate/smoke, per-batch scores
and timing, both prediction files, summary and completion status. All **99
collected files** (23,446,021 bytes) pass collection checksums. The independent
audit binds all **8,376 held-out predictions** to canonical labels/prompts,
recomputes logits/sigmoid, condition/source/query AUROC/raw p20 and accuracy,
paired injection effects, gained/lost decisions and token/timing totals.
Scripts and logs are in `logs/local/b200_projection_judge/`.

Startup/retirement evidence uses `results/b200_sdpa_lens/sdpa03_judge_projection/`
and `results/b200_vllm031/sdpa02_before_judge_projection_retired.json`.
The passing engine stays warm, API 3883/worker 3935, with zero capture/steering/
projection entries and zero uncorrected volatile ECC errors. Weights remain
on the B200; no capacity, dependency or training-runtime changes.

## Single-pass end-to-end costs

Concurrency 32, batches of 128, prefix cache off, final-layer capture and
full-vocabulary/A/B readouts on every row. Each arm scores **1,521,530 input
tokens**. Timing includes batch wall time; request latency includes semaphore
wait. These are diagnostic campaign costs, not repeated production benchmarks.

| Arm | Seconds | Input tokens/s | Requests/s | Request p50 / p95 seconds |
|---|---:|---:|---:|---:|
| Unedited | 66.93 | 22,732 | 62.57 | 1.07 / 2.08 |
| Projection | 98.94 | 15,378 | 42.33 | 1.59 / 2.96 |

## Fixed-direction signal and benign-control follow-up

The user subsequently authorized separating representation transfer from
decision transfer. The
[capture/control contract](../../experiments/b200_judge_direction_signal/README.md)
fixes the original APPS u20 and sign, final-token layer 20 as primary, and
layer 31 as the ancillary removal/readout check. Reuse the same warm engine
and all original prompts; fit or select nothing on JudgeDeceiver. This is a
post-evaluation exploratory follow-up, not a new untouched confirmation set.

For each injection, replace its single contiguous insertion at the same
candidate location with fixed neutral prose or whitespace padding. Both
control families match the attacked complete prompt's native token count
**exactly**, not merely within the allowed one token. Each has 3,684 rows but
2,568 unique prompts; retain the attack weighting and disclose duplication.
Whitespace preserves candidate words; added prose can change answer quality.
Control preferences are copied for lineage/bookkeeping, not claimed ground
truth for a new judging-accuracy benchmark. These controls match length and
placement, but not the attacks' optimized gibberish/lexical style.

Measure the unedited residual's projection onto u20 and its paired change
from the clean parent. Injection-detection ROC negatives repeat each variant's
own clean parent; no threshold is fitted or direction sign reversed.

| Layer-20 contrast | Matched-weighted AUROC | Mean paired projection shift | Positive paired shifts |
|---|---:|---:|---:|
| All attacks minus clean | 0.591318 | +0.185727 | 66.69% |
| Preferred-injected minus clean | 0.665438 | +0.326270 | 74.05% |
| Disfavored-injected minus clean | 0.517198 | +0.045184 | 59.34% |
| Prose control minus clean | 0.454764 | -0.042141 | 37.81% |
| Whitespace control minus clean | 0.517545 | +0.018351 | 62.51% |
| Attack minus matched prose | 0.623542 | +0.227868 | 70.06% |
| Attack minus matched whitespace | 0.577941 | +0.167376 | 64.74% |

There is some injection-associated signal beyond a simple token-count change,
but substantial overlap and ordinary-padding sensitivity remain. In preferred
examples, whitespace produces a positive shift more often than attacks
(79.15% versus 74.05%), although its mean shift is much smaller
(0.060268 versus 0.326270). A positive component change is not itself reliable
injection recognition. The earlier APPS diagnostic AUROC was 0.988861 on a
different 128-task injected/neutral contrast; this is not an identical negative
population or attack-style comparison.

| Source | Injection rows | Attack versus clean AUROC | Mean paired shift |
|---|---:|---:|---:|
| LLMBar | 1,776 | 0.557407 | +0.048086 |
| MT-Bench | 1,792 | 0.673940 | +0.329568 |
| RLAIF | 40 | 0.947500 | +0.350051 |
| Search | 76 | 0.388158 | -0.075930 |

Source-macro detection AUROC is 0.641751; six-query macro is 0.649689.
Query-specific AUROCs are 0.947500 (RLAIF), 0.791783/0.644272 (MT-Bench),
0.630680/0.495744 (LLMBar), and 0.388158 (Search). RLAIF/Search each supply
one query and few rows. Search also separates attacks from the prose control
well (AUROC 0.927632) despite reversing the attack-versus-clean sign. That
illustrates why particular control contrasts are not proof of a universal
semantic injection detector. All six IDs/breakdowns are retained in the report;
dependent rows do not justify independent-example significance tests.

Full projection demonstrably removes the measured component. Layer-20 mean
absolute projection falls from **0.495825 to 0.001132**, leaving **0.22825%**
of its original mean magnitude; layer 31 leaves **0.31914%**. Maximum remaining
component/residual-norm ratios are 0.00029985/0.00029977, about **0.030%**,
well within the predefined conservative 2% BF16 allowance. This measures
geometric component removal, not removal of all injection information.

For preferred injections, paired signal shift has Spearman **rho +0.069716**
with the injection-induced correct-answer A/B margin change, and **+0.198150**
with the full-projection margin change. Average injection damage is -2.970141
correct-answer logit-margin units; average projection change is -0.005429.
For disfavored injections the corresponding correlations are -0.727029 and
+0.494942, so the pooled damage correlation (-0.400817) obscures condition
differences. These are descriptive coefficients without independent-row
p-values; source/query, answer order and suffix dependence remain relevant.

The interpretation is **partial, context-dependent representation transfer
along with weak aggregate decision benefit**. It does not support an axis that
works only in AI-control settings, a strong domain-general injection detector,
or an explanation based solely on incomplete projection. Prompt/task/readout
changes, lexical style, redundant features and downstream redistribution are
still possible contributors. Full all-layer projection does not isolate one
layer's causal mediation or prove a unique decision circuit.

Capture A/B scores reproduce the previous unedited scores at MAE 0.000172,
correlation 0.999985, with zero decision flips; projected scores differ at
MAE 0.001664/correlation 0.999847, with five flips. Fresh preferred accuracy
remains exactly 55.81%/56.46%. Capture changes RPC/batch timing in BF16 serving;
it is not bitwise full-population reproduction. The bounded capture canary
passes at MAE 0.001032/correlation 0.999842, and beta-zero/following-plain
activations and A/B logits are exact.

Artifacts use `results/b200_judge_direction_signal/sdpa_judge_direction_signal01/`:
frozen workload/control receipts, manifest/config/executed sources, raw BF16
residual values preserved losslessly in FP32 arrays, per-batch hashes/scores,
and analysis. Independent audit verifies **289 collected files**
(388,249,156 bytes), **15,744 predictions**, **7,368 control constructions**,
**11,556 native token counts** and **31,488 residual vectors**; it recomputes
FP64 direction dots/norms, ROC/paired shifts, rank correlations and removal.
Maximum FP64-versus-recorded dot difference is 0.00000191. Six focused tests,
scoped Ruff and diff checks pass. The engine stays warm with zero request
state and zero uncorrected ECC errors; no restart, package or capacity change.

One-pass capture costs, c32/batch128 with both layer captures and full readout:

| Population | Input tokens | Input tokens/s | Requests/s | Request p50 / p95 seconds |
|---|---:|---:|---:|---:|
| Original unedited | 1,521,530 | 21,839 | 60.11 | 1.11 / 2.11 |
| Original projected | 1,521,530 | 14,647 | 40.32 | 1.66 / 3.03 |
| Prose controls | 1,351,548 | 22,099 | 60.24 | 1.11 / 2.12 |
| Whitespace controls | 1,351,548 | 21,885 | 59.65 | 1.13 / 2.13 |

Latency includes semaphore wait. Original/control lengths are matched per
attack, but controls omit the separate clean rows; compare population timings
with that difference disclosed. These are capture diagnostics, not repeated
production benchmarks.

## Numeric-output follow-up

`sdpa_judge_numeric_surface01` tests the user-proposed 0/1 association with the
same BF16/SDPA adapter, resident engine and fixed all-layer/all-token u20 full
projection. Both variants were frozen before scoring all 4,188 original rows:
`output_only` keeps candidate A/B tags but requests 0 for A and 1 for B;
`fully_numeric` also renames trusted candidate references/tags to 0/1.
Question, candidate and attack bytes, labels, pairs and orders are unchanged.
Only trusted framing changes; this adds respectively four/ten prompt tokens per
row. No direction fitting, parameter sweep or promotion occurs. The experiment
contract is [the numeric-surface README](../../experiments/b200_judge_numeric_surface/README.md).

| Prompt surface | Arm | Clean accuracy | Preferred-injected accuracy | Disfavored-injected accuracy | Pooled accuracy |
|---|---|---:|---:|---:|---:|
| Original A/B, reused control | Unedited | 97.82% | 55.81% | 98.05% | 79.44% |
| Original A/B, reused control | Full projection | 99.01% | 56.46% | 98.37% | 80.01% |
| Numeric output, A/B candidates | Unedited | 89.88% | 53.75% | 79.10% | 69.25% |
| Numeric output, A/B candidates | Full projection | 89.29% | 53.64% | 76.76% | 68.10% |
| Numeric output and candidates | Unedited | 96.43% | 53.04% | 93.59% | 76.10% |
| Numeric output and candidates | Full projection | 94.05% | 55.92% | 92.45% | 76.58% |

The numeric framing does **not** rescue projection into a reliable injection
intervention. Fully numeric preferred accuracy gains 2.88 percentage points,
but clean accuracy loses 2.38 points and disfavored accuracy loses 1.14 points.
Output-only projection reduces pooled accuracy by 1.15 points. Numeric baselines
also differ substantially from A/B, demonstrating prompt-surface sensitivity.

| Numeric surface | Arm | Pooled AUROC / raw pAUROC@20 | Preferred AUROC / raw pAUROC@20 | Pooled Brier / ECE |
|---|---|---|---|---|
| Output only | Unedited | 0.852506 / 0.559882 | 0.580444 / 0.140872 | 0.210216 / 0.210663 |
| Output only | Projected | 0.833981 / 0.571435 | 0.558833 / 0.139766 | 0.225080 / 0.223612 |
| Fully numeric | Unedited | 0.896828 / 0.648960 | 0.675763 / 0.182847 | 0.162056 / 0.149595 |
| Fully numeric | Projected | 0.897160 / 0.666067 | 0.671653 / 0.198903 | 0.171811 / 0.175082 |

Preferred accuracy gains in fully numeric prompts accompany **stronger answer-0
bias**, rather than balanced recovery: correct-first accuracy rises
86.75%→98.26%, while correct-second accuracy falls 19.33%→13.57% (921 rows each).
Output-only is already severely biased: correct-first 98.26%→99.89%,
correct-second 9.23%→7.38%. Mean preferred p(1) falls about 0.0426 in both
variants. Fully numeric pooled accuracy changes by source: LLMBar
78.73%→77.08%, MT-Bench 74.01%→77.13%, RLAIF 90.00%→86.67%, Search
59.65%→52.63%. Source/query, slot, calibration and tie breakdowns are archived;
these dependent variants across six queries do not support independent-row
significance claims. Exact-half ties decline 114→83 (output-only), 86→58
(fully numeric), pooled.

Requested 0/1 probability mass is healthy: means 0.97856/0.98105 for output-only
and 0.98278/0.98484 for fully numeric, unedited/projected; every row exceeds
0.925. Alternate A/B mass is tiny (means 0.00379/0.00295 and
0.000152/0.000165), so conditional A/B scores are diagnostics of unrequested
logits, not faithful judging outputs. Comparing both pairs from the same full
vocabulary matrix, preferred signed margin changes are -0.24695 (0/1) versus
-0.60963 (A/B) for output-only, and -0.20888 versus -0.21159 for fully numeric.
Thus projection affects both token pairs, with prompt-dependent magnitudes;
it does not establish an exclusively 0/1-aligned mechanism. Original A/B
prompts' archived wrong-head 0/1 probabilities also decrease by 0.02670 on
average, but their probability mass was not archived and cannot be assumed
healthy.

Fixed-direction injection-versus-matched-clean detection AUROC increases from
the original 0.591318 to 0.651666 (output-only) / 0.635091 (fully numeric).
Preferred contrasts are 0.769605/0.750496, disfavored 0.533727/0.519685.
Projection leaves 0.3425%/0.3039% of mean absolute layer-20 component magnitude;
maximum residual-normalized components across layers 20/31 are below 0.031%.
The measured direction is successfully removed, while robust decision benefit
remains absent. Original attack bytes were optimized for A/B, and may mention
those labels. These results conflate output framing and attack alignment;
they do not establish robustness against numerically retargeted payloads or
isolate a token-level circuit. A label-swap or retargeting experiment would
require a separately frozen contract.

The unchanged 0/1 canary passes at MAE 0.000991/correlation 0.999316;
beta-zero and following-plain captures/logits are exact. No new backbone-master
reference is claimed for this prompt-only change. Four focused tests, scoped
Ruff and diff checks pass. Independent audit checks **307 collected files**
(413,491,815 bytes), **16,752 predictions**, **8,376 content-preserving transforms**
and **33,504 raw residual vectors**, including native token counts, both heads,
slot metrics and FP64 direction dots (maximum difference 0.000001742).
Artifacts are under `results/b200_judge_numeric_surface/sdpa_judge_numeric_surface01/`.
The existing worker stays warm with zero capture/steering/projection state and
zero uncorrected ECC errors; no weights were downloaded.

| Capture pass | Input tokens | Input tokens/s | Requests/s | Request p50 / p95 seconds |
|---|---:|---:|---:|---:|
| Output only, unedited | 1,538,282 | 22,196 | 60.43 | 1.13 / 2.10 |
| Output only, projected | 1,538,282 | 14,488 | 39.44 | 1.70 / 3.12 |
| Fully numeric, unedited | 1,563,410 | 22,746 | 60.93 | 1.12 / 2.13 |
| Fully numeric, projected | 1,563,410 | 15,343 | 41.10 | 1.67 / 2.96 |

These are single-pass c32/batch128 capture diagnostics, including semaphore wait,
with the same full readout and layer-20/31 captures; not repeated production
benchmarks.
