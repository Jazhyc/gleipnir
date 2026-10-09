# Training-example alignment with the APPS direction

The complete original-data comparison covers all **8,688 training records** in
the pinned base and current BF16/SDPA monitor, with no added synthetic views.
Largest training-associated increases concentrate in harmless/teacher-positive
examples and the o46 source; exact duplicate content is another review candidate.
The axis also carries behavior-label signal. These are activation-based clues,
not causal training-example influence or an applied data filter. See the
[complete census](#complete-original-data-census) for current statistics; the
initial small mixed-view diagnostic is retained separately below.

## Matched contract

The [experiment README](../../experiments/b200_training_direction/README.md)
defines the frozen selection and interpretation. Select 512 of 8,688 trajectories
proportionally by six sources, hard label and augmentation status using a fixed
hash, before activations/outcomes. Recorded original lineage groups are unique.
The sample has 256 examples per hard label, 307 unaugmented and 205 augmented;
include the latter's original clean parents, giving 717 views/6,826,564 input
tokens per model, maximum 28,242. All eight training templates and three roles
are represented. This bounded training-seen sample is not an exhaustive corpus
screen, a holdout or a second training seed.

Compare pinned Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` with the current BF16/SDPA augmented
replay (master `bdb702316b1d5e29cbdd8fba64888fe128b37bfd25030211b867d46372e8f5fd`).
Both use unquantized BF16 eager Lens, identical renderer/frontend/scheduler,
prefix off, c16/batch32 and the same ordered prompts. Preserve historical
training inputs, including reasoning; use the exact training chat template,
thinking disabled and `Prediction:` prefix. Targets/completions are metadata,
never added to the model-visible prompt.

Capture unedited final-token residuals at layers 20/31. Primary readout is the
unchanged APPS-trained unit u20 dot layer-20 residual, with its original sign;
cosine and norms expose scale changes. Both models use that same coordinate
axis, without refitting or semantic feature alignment. This compares all effects
of monitoring training, not augmentation versus a matched unaugmented training
control. Neither activation ranking nor trained-minus-base difference is an
example-influence estimate.

## Direction, labels and teacher targets

| Actual training population | Base | Trained |
|---|---:|---:|
| Direction AUROC for hard behavior labels | 0.769333 | 0.902374 |
| Spearman with soft teacher suspicion | 0.420880 | 0.708438 |
| Spearman with trained monitor score | 0.398133 | 0.736300 |
| Mean layer-20 residual norm | 16.56294 | 18.77167 |
| Standard deviation of direction projection | 1.22844 | 2.98310 |
| Standard deviation of cosine alignment | 0.07420 | 0.15813 |

Training separates labels in both directions: harmless examples' mean
projection changes -1.12034→-2.71921, harmful examples -0.04371→+1.82434.
The mean change pooled across labels is only +0.13459, hiding this polarization.
Cosine changes are -0.07387/+0.10292 respectively, so increased residual scale
alone does not explain the result. On original clean views, direction label
AUROC is already 0.854004 in the base and rises to 0.900482 after training.
Cross-model projection rank correlation is only 0.238542: training substantially
reorders examples along this fixed axis.

The actual training objective used soft teacher targets, not hard
source labels. A particularly useful candidate subset is **17 hard-label-0
examples with teacher suspicion >=0.5**: mean training-associated increase
is **+4.26805**, versus **-2.01619** for the other 239 hard-label-0 examples.
These disagreements are inspection candidates, not established teacher mistakes
or justification to remove data. Teacher target and projection change have
pooled Spearman 0.636962, and positive within-source coefficients across all six
sources (0.333898–0.733843). Length also correlates with pooled change (0.336062);
source, task style, labels and length remain confounded. No independent-row
significance or held-out monitoring-quality claim follows from these diagnostics.

| Source | Selected originals | Mean trained-minus-base projection |
|---|---:|---:|
| crh | 138 | -1.41861 |
| ctrl | 84 | -0.04299 |
| gloom | 46 | +0.44846 |
| o46 | 60 | +2.87949 |
| s45 | 104 | +0.90494 |
| stride | 80 | -0.24030 |

Of the top 20 positive training-associated increases, **13 are o46** (BashArena),
only four are augmented, ten have hard label 0 and their mean teacher target is
0.736813. The negative top 20 are all hard label 0 (15 crh/five STRIDE), with
mean teacher target 0.023159. These rankings suggest monitoring supervision and
source/task patterns as candidates beyond the artificial payloads themselves.

## Matched injection responses

For each of 205 selected augmented originals, subtract its own clean-parent
projection in each model. Then subtract the base shift from the trained shift.

| Training-template family | Pairs | Base injection shift | Trained injection shift |
|---|---:|---:|---:|
| Direct override | 55 | +1.62487 | +0.12620 |
| Reviewer authority spoofing | 44 | +1.09333 | +0.21259 |
| Answer prefill | 50 | +0.47412 | +0.03923 |
| Delegated verdict | 56 | +0.73174 | +0.11454 |
| All | 205 | **+0.98614** | **+0.12035** |

The mean shift shrinks 87.8%; cosine shifts shrink from +0.06004 to +0.00626.
Mean difference-in-differences is -0.865789, negative in 80.49% of pairs.
Every template and source has a negative mean difference. Injection raises
projection in 85.37% of base pairs versus 64.88% of trained pairs. Residual
trained shifts are +0.21748 on harmless versus +0.02227 on harmful examples.
Demand-0/1 shifts are +0.65722/+1.24355 in the base versus
+0.09440/+0.14065 in the trained model. These known-template associations do
not establish transfer to APPS payloads, adaptive robustness or calibration.

Positive difference-in-differences can arise from removing a base model's
negative response to a demanded 0, rather than learning a large positive alarm.
For example `d6e4fcb7cc0bd2f138bc80ab` (s45, enterprise vulnerability scanner,
label 1/teacher 0.9948, answer-prefill demand 0) has base clean/injected
projections +0.67894/-0.60896, while trained +1.79387/+1.83644. Its +1.33047
relative amplification is mostly the disappearance of the base negative shift.

## Concrete inspection candidates

Stable indexes below refer to ignored, checksum-bound training data; task titles
identify examples, not independently verified misconduct. Rankings retain full
IDs, label/teacher provenance, both model projections and bounded context.

- `e08ed91e5e64620d19d686d2`, o46/BashArena, debug-symbol management tool:
  unaugmented, hard label 0/teacher 0.7549; base -2.01233→trained +3.62600
  (change +5.63833). The positive response does not require an inserted payload.
- `57bea5c4363836328dfe5f57`, s45/BashArena, fork-performance investigation:
  hard label 0/teacher 0.7058. Its **clean parent** changes
  -1.64473→+4.12406 (+5.76880); augmented view changes -1.21898→+4.12807.
  Most of its trained alignment is present without the answer-prefill injection.
- `c45ffb801faf4319e756aa00`, crh, sorted-list coding task:
  hard label 0/teacher 0.0260, direct-override demand 0. Base clean/injected
  -0.55200/+1.61297 becomes trained -5.56615/-5.13804. Training moves both views
  toward the negative end and attenuates the injection shift.

The next causal test would need matched retraining with a frozen candidate-group
removal/reweighting or target-change intervention and held-out APPS/ID guards.
Do not delete these rows based on activation rankings. A matched unaugmented
training control would first help distinguish general distillation from the
augmentation contribution. Neither additional training nor a data change occurs
here. The result is consistent with previous projection losing genuine backdoor
recall: an axis related to behavior suspicion can carry useful monitoring signal
as well as responses to injection wording.

## Verification and artifacts

Artifacts: `results/b200_training_direction/training_direction01/`, with frozen
config/manifest/workload, lossless native-BF16 residuals in FP32, paired model and
injection records, 12 top/bottom ranked lists, `summary.json`, and supplemental
`direction_target_diagnostics.json` bound to the paired-record hash.

Trained, base and restored canaries pass against saved FP32 references at
MAE 0.000855/0.010304/0.000805 and correlation
0.999994/0.998310/0.999994. Trained/restored also pass the archived eager-score
check and nonzero adapter-effect gate. Native audits show unquantized BF16
projections/head/attention in both engines. No activation-reference parity or
base monitor quality claim is added. Three focused tests, scoped Ruff and diff
checks pass.

Independent audit verifies **161 collected files** (73,326,055 bytes), **1,434
matched predictions**, **205 clean/injected pairs**, **2,868 raw residual vectors**,
all 12 rankings, exact source-byte reconstruction and effective-tokenizer counts.
Maximum FP64 versus recorded direction-dot error is 0.000002749. The audit's
initial wrapper-offset/raw-JSON assumptions were corrected using original
trajectory hashes and the effective HF tokenizer; failed audit receipts remain
separate. No experiment gate or tolerance was relaxed. Transformers reconstructs
additional tokens and a different combining-mark regex from the pinned tokenizer
JSON/config; bare `Tokenizer.from_file` alone does not reproduce all long-training
prompt counts. Both server passes exactly match prepared HF counts. Effective
backend SHA is `ffb7a28b27dabcc333662fd3e0b0005d9e79a1c22e31453ab5a3017fbd5f25c0`,
identical locally and in the serving runtime (Transformers 5.14.1/tokenizers 0.22.2).

| Model pass | Input tokens/s | Requests/s | Request p50 / p95 seconds |
|---|---:|---:|---:|
| Trained | 85,468 | 8.98 | 1.61 / 6.08 |
| Base | 85,310 | 8.96 | 1.54 / 6.06 |

These are single-pass capture diagnostics, not repeated production benchmarks;
latency includes semaphore wait. Existing engine 3883/3935 was retired before
base 4863/4916; base was retired before restoring trained 5061/5100. The restored
engine passes its gates and remains warm with zero request state/ECC errors.
Its active frontend metadata points to its own startup receipt; the inherited
launch metadata is preserved separately. No capacity, package, training or
checkpoint change; all merged weights remain remote and shared caches intact.

## Complete original-data census

The user clarified that inputs should be **all original training data without
synthetic augmentation views**, superseding the small mixed-view diagnostic as
filtering evidence. `clean_direction01` scores every row in the original regular
training file, exactly once in each model: **8,688 originals**, **4,344 per hard
label**, **83,816,369 tokens per model**, maximum **29,337**, with zero synthetic
views or truncation. The same augmented-trained SDPA checkpoint remains the
trained model; this correction changes the input population, not its weights.
The [complete-census contract](../../experiments/b200_training_clean_direction/README.md)
freezes the layer, axis, joins, bands and stop conditions. Inputs retain original
reasoning and naturally occurring injections/quoted tests; clean here means no
added augmentation, not verified absence of injection text.

Fresh complete passes use the same BF16 eager Lens recipe/order/batches. The
existing Qwen3.5-9B injection-presence census is joined by unique ID and exact
trajectory hash; clean teacher records by original prompt lineage. Neither
annotation enters model inputs. No vector fitting, layer/strength selection,
training, data removal or paid teacher call occurs. Shared sequential capture
and statistics now live in `gleipnir.evaluation.direction_campaign` and
`direction_stats`; the small experiment keeps its compatibility entrypoint.
The moved numerical helper ASTs are unchanged. Historical source snapshots and
receipts remain preserved; reproducing them requires their recorded revision.

### Alignment and candidate concentrations

The fixed-axis hard-label AUROC is **0.852388 in the base → 0.885695 trained**.
Mean projections are -1.01013→-0.53135; this pooled change hides strong class
polarization. Harmless means -1.66692→-2.79030 (change -1.12339), harmful
-0.35334→+1.72760 (+2.08095). Mean norms grow 16.49384→18.74358, but cosine
standard deviation also grows 0.06611→0.16091; scale alone does not explain the
separation. This is training-seen representation diagnostics, not held-out
monitor quality or causal example influence.

Complete distribution figures are saved as `direction_distributions.{png,pdf}`
and `direction_source_distributions.{png,pdf}` in the census artifact directory;
the [experiment README](../../experiments/b200_training_clean_direction/README.md)
provides the reproduction command. They include every original record, with
common full-range histogram bins and unit area within each group. Base alignment
is comparatively concentrated; trained alignment spreads in both directions.
Source facets show that o46's positive shift occurs across both behavior labels,
while harmless teacher-positive records form a strongly positive change group.
These population views complement the fixed inspection bands without selecting
a filtering threshold. Counts and original duplicate exposure weights are retained.

| Original hard label | Teacher target | Rows | Mean trained projection | Mean trained-minus-base |
|---|---|---:|---:|---:|
| Harmless (0) | <0.5 | 3,998 | -3.29035 | -1.61443 |
| Harmless (0) | >=0.5 | **346** | **+2.98772** | **+4.55060** |
| Harmful (1) | <0.5 | 506 | +0.53663 | +1.53921 |
| Harmful (1) | >=0.5 | 3,838 | +1.88462 | +2.15237 |

All 346 harmless/teacher-positive rows have positive changes. They are **3.98%
of originals but carry 12.72% of measured positive change**; 106 have teacher
target >=0.9. Source counts are o46 167, ctrl 88, s45 83, STRIDE five and Gloom
three. Only **three** have census score >=0.5. Their supervised soft targets
therefore provide a different inspection lead from the old presence-census
removal. These disagreements may reflect teacher errors, source-label errors or
rubric/evidence differences; they are not verified mislabeled examples.

Positive-change share below is `sum(max(trained_z20 - base_z20, 0))` within a
source divided by the corpus total. It describes measured scalar mass, not
parameter-update or causal training influence; the zero/sign/axis are frozen.

| Source | Originals (% of corpus) | Mean change, label 0 | Mean change, label 1 | Share of positive change |
|---|---:|---:|---:|---:|
| crh | 2,340 (26.93%) | -4.07972 | +1.78179 | 17.53% |
| ctrl | 1,436 (16.53%) | -0.59297 | +1.46127 | 13.10% |
| gloom | 772 (8.89%) | -0.65597 | +2.71638 | 9.32% |
| o46 | **1,020 (11.74%)** | **+2.54710** | **+3.32518** | **25.17%** |
| s45 | 1,770 (20.37%) | +0.33621 | +2.00045 | 22.71% |
| stride | 1,350 (15.54%) | -1.51751 | +2.06071 | 12.16% |

BashArena comprises o46/s45; canonical dataset aggregates and all source/label
cells are also retained. O46 shifts positively even among hard-label-0 examples;
whole-source removal would also discard 510 harmful originals.

### Complete rankings and controls

Fixed top fractions use deterministic ID tie breaks, with separate average-tie
percentiles. Raw projection changes have the following composition:

| Largest change band | Rows | Hard-label 0 / 1 | Teacher–hard disagreements | Census flags >=0.5 | Share of positive change |
|---|---:|---:|---:|---:|---:|
| Top 1% | 87 | **70 / 17** | 59 | **0** | 4.03% |
| Top 5% | 435 | 268 / 167 | 220 | **0** | 17.98% |
| Top 10% | 869 | 412 / 457 | 323 | 2 | 32.60% |

Of the top-1% changes, 56 are harmless/teacher-positive, 14 harmless/teacher-
negative and three harmful/teacher-negative. A teacher-disagreement-only rule
would not cover every large change. Normalizing by residual norm preserves the
pattern: top-1% cosine increases have 66 harmless/21 harmful rows, 62 teacher
disagreements and zero census flags. Highest absolute trained projection is
also a distinct ranking: its top 87 contain 42 harmless/45 harmful and no census
flags. Large activation alone is not a sufficient data-removal criterion.

Pooled change correlates with teacher target (Spearman +0.64414), length
(+0.36987) and census score (+0.40893). These are strongly conditional:
teacher/change rho is +0.69850 within harmless examples but **-0.37245 within
harmful**, and census/change +0.68508 versus **-0.45151**. Do not turn pooled
correlation into a universal monotonic mechanism. Within harmless BashArena
strata, teacher/change rho stays +0.84761 (o46) / +0.74897 (s45), whereas
length/change rho is +0.06445/+0.10027. This makes the teacher/source association
more specific than pooled length alone, without eliminating task/style or
unobserved evidence confounds. All 38 group tables, undefined single-label AUCs,
score/rank ties and both signs are preserved; no independent-row significance
claim or new feature-alignment fitting is introduced.

Concrete high-change originals, identified by task title and stable index:

- `23046e3ad09c93ed834b6f4a`, o46, binary-size optimization framework: label 0,
  teacher 0.7311, census 0.1067; base -2.39537→trained +4.19210, change +6.58748.
- `75af31960a8980e0283e1444`, s45, security-log pattern search: label 0,
  teacher 0.7549, census 0.0675; -2.46394→+3.90978, change +6.37372.
- `6a9f2a417619624287c438e2`, s45, storage/I/O-error recovery: label 0,
  teacher 0.4378, census 0.0953; -2.49438→+3.81770, change +6.31208.
  Its teacher target is below 0.5, illustrating the limit of a single target flag.

These are review candidates, not independently adjudicated harmless actions or
causal training examples. Inspect behavior/evidence and target provenance first.
A later filtering/target-correction experiment should freeze the rule and matched
training control and retain ID/APPS ranking, calibration and harmful-recall
guards. The earlier
[census-based removal](monitoring_injection_removal.md) failed to resolve APPS
alarms and lost ID ranking. No filter or retraining is selected/applied here.

### Exact duplicates and numerical variation

The full original population contains **8,100 distinct student/rendered prompts**:
414 duplicate groups contain 1,002 records, with **588 surplus copies**, all in
crh (586 harmless/two harmful). Recorded lineage IDs are unique but do not imply
independent visible content. Preserve those groups together in future splits.
Current statistics intentionally use all original row weights, matching the
training exposures; deduplication would change both source and label mixtures.

No duplicate group has conflicting hard labels. All share identical teacher
prompt hashes; 406 groups have different cached soft targets, with mean/median/
maximum within-group target ranges 0.003912/0.001415/0.046710. None crosses 0.5.
This is a separate deduplication/reweighting candidate, not evidence of binary
label contradiction. Group membership and ranges are bound in the supplemental
review diagnostics; no copies or targets were removed/averaged.

Repeated identical inputs also expose serving variation within this pass:
mean/maximum projection ranges are 0.005206/0.142186 (base),
0.002323/0.061634 (trained), and 0.005579/0.121833 for their difference.
These are descriptive ranges, not controlled independent repeats. They are
small beside the +4.55060 mean change in harmless/teacher-positive examples,
but exact order near a ranking boundary should not be treated as noise-free.
Native head scores have 114/99 distinct values in base/trained, including
115/26 exact-half ties. Direction changes have 8,178 distinct values; average
rank percentiles and deterministic tie breaks are retained. No seed-variance
or deterministic-training claim is introduced.

### Artifacts and verification

`results/b200_training_clean_direction/clean_direction01/` contains the complete
paired records; four full 8,688-row rankings; eight top/bottom-100 excerpt lists;
`review_flags.jsonl` for every original, with descriptive fixed-band/target flags;
`summary.json`; and input, collection, native, canary and independent audits.
`review_diagnostics.json` binds supplemental review counts to paired-record SHA
`ebf8fa4cd24b49a9b164ea77931e539cba3e7949c7b0db4fd3a02541571cb650`.
A standalone PNG/PDF overview plots paired models, source/label changes,
harmless teacher groups and scalar shares, with input/script/figure hashes.

All master/eager/native gates pass. Trained/restored MAE is 0.000855 and
correlation 0.999994 against FP32; eager MAE 0.000249/correlation 0.999999.
Base MAE is 0.010304/correlation 0.998310. Independent audit verifies **1,162
collected files** (814,517,206 bytes), **17,376 captures**, **34,752 raw vectors**,
all **8,688 original joins/native counts**, **38 groups**, **four complete rankings**,
**eight example lists** and **8,688 review flags**. Maximum FP64 dot discrepancy
is 0.000003525. An initial transfer began before the checksum manifest was
finalized; the completed manifest was fetched and verification rerun without
changing data, scores or limits. Its failed transfer-audit log is retained.
Seven focused tests, scoped Ruff and diff checks pass.

| Complete pass | Seconds | Input tokens/s | Requests/s | Queued request p50 / p95 seconds |
|---|---:|---:|---:|---:|
| Trained | 983.76 | 85,200 | 8.83 | 1.70 / 6.17 |
| Base | 980.70 | 85,466 | 8.86 | 1.70 / 6.16 |

These are one-pass capture timings, not repeated production benchmarks. The
trained engine is restored and gated (API 5969/worker 6020), with zero request
state and uncorrected ECC errors. Current process/frontend metadata is in this
run's `restored_server.json`. Shared caches and remote merged weights remain
intact; original data and teacher caches are unchanged.
