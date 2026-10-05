# Full regular 4B replication with native FP4 MLPs

Decision date: 2026-10-05. The user requests a complete training run followed by
ID evaluation and selects the original regular 4B run: 8,688 monitoring rows,
LR 5e-5, one epoch. This follows the explicit selection of native FP4 MLPs as
the B200 execution default. The corrected epoch and canonical ID evaluation
complete on 2026-10-06.
This one-seed comparison does not establish equivalence with BF16.

## Frozen comparison

Use `experiments/b200_fp4_full_training/config.yaml` and the named
`qwen35_4b_b200_fp4_mlp` profile. Preserve the original regular prompt, Kimi soft
targets, seed 0, initial FP32 master adapter, rank 128/alpha 256, AdamW schedule,
logical batch 32, packing budget 16,384, no checkpointing or truncation, and
final one-epoch selection. Expect 272 updates and 83,816,369 input tokens.
Frozen source checksums, lineage holdouts and the executed source archive bind
the run to the original inputs; no prompts or target distributions change.

Reuse the original regular control and its cached ID scores. Its training used
BF16 MLPs and SDPA; the candidate combines native FP4 MLPs and BF16 FA4. Report
their differences as a combined recipe comparison, not an isolated FP4 effect.
Historical startup/cache conditions also differ. No additional full BF16
control, ID tuning, alternate epoch selection, OOD evaluation, publication or
quality promotion is part of this request.

Evaluate all 3,012 canonical CoT-removed ID examples using the regular prompt.
The final FP32 master adapter is rebased for BF16 serving. Require a fresh,
bounded master-to-vLLM score-parity receipt and a nonzero adapter effect before
scaling evaluation. Report ranking metrics, calibration, thresholds, ties and
per-source results against the exact same cached control population. One seed
does not establish equivalence.

## Execution and retained worker

Use the existing US-NC-2 B200 pod `i243nsg10usytq`, network volume `ixbh81vf9c`,
and populated shared compiler/kernel caches. No capacity is created or
terminated. The old resident timing worker, PID 11905, supports only twenty-step
trials; preserve its state/logs and replace that process with the ordinary full
Trainer through a retention wrapper. The initial slow pipeline PID 16069
launches trainer PID 16214; the corrected run retains trainer PID 67332 as
recorded below. After training, keep the model, native plans and initial CPU FP32 master
copies resident, release optimizer state, and park the worker for future work.

Reuse checksum-bound startup validation from
`results/b200_mlp_gemm/warmed03/causal_adapter/training_metadata.json`, SHA256
`14ab15279bb8895cf32353117d5c1cf957ad45d2b0ca9d7205067db27d77edeb`.
Retain historical failed strict diagnostics and explicit selected-finite
acceptance. Continue finite/missing-gradient checks on every update. No model
warmup replay, packing canary or longest-batch numerical preflight is repeated.
Adapter-specific serving parity remains required.

Artifacts live in `results/b200_fp4_full_training/`; logs in
`logs/runpod/b200_fp4_full_training/`; input manifest in
`data/b200_fp4_full_training/manifest.json`. `execution_contract.json` binds
executed sources, command and manifest. `worker_replacement.json`, `launch.json`
and `worker.json` record process replacement and current state. No in-chat
scheduling tool is available; active-turn checks cannot promise agent follow-ups
after the turn ends.

## Full-corpus cache costs

The original epoch contains 4,550 physical batches with 3,086 unique packed token
counts. Native plans are keyed by exact token count, so the short warmed timing
cohort does not describe first-use overhead across this population. The first
candidate update takes about 397 seconds including compilation; early subsequent
updates take roughly 85–113 seconds. These are interim wall times, not completed
epoch throughput or the previously reported warmed 3.678-second cohort result.

A bounded compile-only probe shows persistent-object cache hits and a remaining
first-use miss. Use an optional 16-process helper to populate the unchanged
native plans in spare CPUs, without loading or replaying a model or executing
GEMMs. Preserve shared cache namespaces and atomic upstream cache publication.
Archive helper source, future shapes, cache hit/miss counts and execution time
separately; account for this concurrent work when reporting full-run timing.
The first parallel pilot incorrectly rejects the backend's one-byte workspace
marker. Preserve that failure; the corrected helper permits only that marker,
requires zero fused workspace, and checks GPU headroom for compiler contexts.

Full training and ID results remain pending. Collect and verify final artifacts
before drawing quality or practical full-corpus performance conclusions.

## Full-corpus throughput regression and runtime-shape fix

The first attempt was paused after 43 of 272 updates with its live state intact.
The latest completed update takes about 45 seconds, after earlier updates of
roughly 80–113 seconds; this is substantially slower than the historical control
(3,916 seconds for the whole epoch). The user expects roughly an hour. Do not
extrapolate the 3.678-second short-cohort measurement to this token distribution
or accept a multi-hour run without investigating the regression.

A compile-only CPU profile of the fourteen packed shapes at historical update
44 takes 23.980 seconds to construct 56 native plans, despite 168 compiled-object
cache hits and zero misses. Repeated graph construction, module loading, template
reads and runtime-library discovery dominate this bounded diagnostic. This is
not a full-training trace, and does not establish exact end-to-end percentages.
Another bounded probe takes 7.275 seconds to compile/load FP4 conversion and
row-scaling variants for the same shapes. The earlier compile-ahead helper
populated GEMM objects, not these Triton variants. Its successful 747-second run
therefore did not resolve the underlying per-shape overhead.

NVIDIA's existing compiled GEMM already supports symbolic runtime M. Reuse one
fused-descaling plan per device/K/N geometry, retaining all geometry, scale-blob
and native launch guards. Keep the fixed-M direct API for historical probes.
Pass conversion group counts and row-scale lengths at runtime instead of
specializing on each count. Preserve K, hardware conversion, fused BF16 rounding,
tile configuration, FP32 adapter masters and all experiment controls.

The bounded native proof covers four projection orientations at M=129, 5,047,
16,322 and 29,337: all sixteen outputs are bitwise identical to fixed-M plans,
with the same NVIDIA tile configuration and no refused launches. A separate
conversion proof covers all three activation widths at seven row counts,
including empty-valued rows and 128-row boundaries: packed codes, scale bytes
and FP32 inverses match bitwise in all 21 cases. Each width uses one compiled
conversion kernel across these lengths. Generic row scaling matches bitwise
at the same seven lengths with one compiled variant. Record these receipts and
source hashes separately from the reused historical startup receipt; unchanged
model canaries are not claimed to have passed again. Full-run throughput and
ID quality remain unmeasured for this fix.

Applying the Python cache-key change requires a process restart. The first
attempt has no intermediate disk checkpoint; a replacement must restart from
the frozen original initialization, preserving the slow attempt's source,
logs and diagnostic receipts. Reuse the same disk caches and allocated B200.

The replacement pipeline PID 66650 and trainer PID 66778 start on the existing
B200 at Unix time 1791235348.779. The slow attempt and logs are archived under
`results/b200_fp4_full_training_slow_attempt01/` and
`logs/runpod/b200_fp4_full_training_slow_attempt01/`. Its 43 updates have no
resumable checkpoint and are not counted toward the replacement epoch.
`runtime_shape_validation.json` binds the 16 GEMM, 21 conversion and seven
row-scale cases plus current source checksums; the new execution contract
records that receipt's hash. Sixty-one focused tests and Ruff pass.

That replacement stops before any update: the original startup source guard
correctly rejects the changed MLP source. Preserve this failure under
`b200_fp4_full_training_source_guard_attempt02` results/log paths. The source
binding now retains historical hashes separately and pins the new hashes to
the 44-case bitwise proof; startup metadata records that separate equivalence
evidence without relabeling old canaries. The same source check runs before
model loading in the campaign launcher. A bounded receipt read passes with
no repeated model arithmetic, and 62 focused tests pass. The corrected pipeline
PID 67204 and trainer PID 67332 restart from the same frozen controls at Unix
time 1791236020.868; throughput and ID remain pending.
The per-shape compile-ahead helper is unnecessary for the runtime-M baseline.

The corrected run reaches update 8 with exactly four native plans and 64 packed
weight pairs. Its first update takes 293 seconds including initial compilation;
updates 2–8 take 12, 10, 11, 8, 10, 10 and 12 seconds (one-second log resolution,
10.43-second mean). All eight logged losses and gradient norms match the archived
slow attempt exactly. This is encouraging full-corpus evidence, not the warmed
short-cohort benchmark or a completed-epoch result. The warmed short cohort
already cached its exact-M plans and conversion variants, so this removal of
first-use overhead should not imply a similar gain there. Generic conversion
may still change steady performance slightly; it has not been timed separately.

## Completed training, 2026-10-06

The corrected epoch completes all 272 updates over 8,688 monitoring rows and
83,816,369 input tokens, with no padding, truncation or missing/nonfinite
gradients. Training-loop runtime is 2,779.6297 seconds (46 minutes 20 seconds),
versus the historical regular control's 3,916.0600 seconds (65 minutes 16 seconds):
29.0197% less time, or 1.4088 times the throughput. This includes the first
293.159-second update's preparation. The remaining 271 logical updates average
8.9631 seconds (median 8.9336, range 3.2086–12.8333); this timer excludes some
Trainer/logging overhead. Do not compare this full-corpus population directly
with the smaller warmed timing cohort.

The successful pipeline takes about 53 minutes 14 seconds through validated
adapter export, including input preparation, model loading, training and export.
The failed earlier attempts and subsequent serving reference/ID evaluation are
separate costs, preserved in their own artifact trees. No additional training
control or per-shape compile-ahead helper runs during the corrected epoch.

Mean training loss is 0.2439715244 versus 0.2331215996 for the historical control,
a descriptive increase of 0.0108499249 (4.65%). The first 43 logged losses and
gradient norms match the archived slow attempt exactly; this is a check of the
runtime-shape fix, not an FP4-to-BF16 quality comparison.

The final causal master SHA256 is
`8dbc1a2e5445ca8807907692b64a59f637e00958d0c89f0d4182930cffef866c`;
the rebased serving artifact SHA256 is
`d13be8b249129269cec572e36e2006a955fa56dec5a015bd93f35bd876c62751`.
Both preserve all 256 FP32 adapter tensors (169,869,312 elements). Their mapped
tensor values are bitwise equal and finite; serving uses BF16 compute rather
than changing the archived adapter dtype. Local collection matches both hashes.
Fresh score parity and canonical ID results are completed below.

Worker PID 67332 remains resident and idle, retaining four native plans and 64
packed weight pairs (2,548,040,192 packed bytes); GPU memory falls to about 15 GB
after gradients and optimizer state are released. Record final timing and
collection receipts under `results/b200_fp4_full_training/`.

The fresh final-adapter serving canary passes on twenty training-source rows:
adapter correlation 0.9997727794 and mean absolute score difference 0.0035214861;
base correlation 0.9984131854 and mean difference 0.0106879820. Canonical ID
evaluation then advances in the same vLLM engine. Before its final report,
inspection finds that the base serving environment excludes the training-only
FA4/cuDNN distributions and uses older Cutlass/TVM versions. The summary CLI
therefore delegates artifact comparison to a child in the pinned training
environment, preserving all runtime/source/provenance checks. Twelve focused
tests and Ruff pass. Archive the summary-only source patch separately from the
already-executed training/scoring sources; no arithmetic or evaluation population
changes, and no failed summary run or waived check is claimed.

## Completed canonical ID evaluation, 2026-10-06

All 3,012 frozen examples complete with finite scores, unique IDs and unchanged
source/label/prompt membership: 2,066 Gloom (1,035 negative/1,031 positive) and
946 STRIDE (369 negative/577 positive). The checksum-bound summary recomputes
both populations against the same inputs; no ID tuning or alternate checkpoint
selection occurs.

| Macro metric | Historical BF16/SDPA | Native FP4/FA4 | Difference |
| --- | ---: | ---: | ---: |
| pAUROC@20 | 0.846273 | 0.886090 | +0.039817 |
| AUROC | 0.951394 | 0.965927 | +0.014533 |
| Brier | 0.089216 | 0.086493 | -0.002723 |
| Balanced accuracy | 0.889656 | 0.894161 | +0.004505 |
| Recall | 0.830638 | 0.810651 | -0.019987 |
| False-positive rate | 0.051326 | 0.022328 | -0.028997 |

Threshold diagnostics use the fixed 0.5 threshold. Ranking improves in both
sources, while the Gloom threshold becomes more conservative: fewer false
positives but lower recall. These are descriptive combined-recipe differences,
not an isolated FP4 benefit or a statistical equivalence/promotion result.

| Source | pAUROC@20, old → new | AUROC, old → new | Brier, old → new | Recall, old → new | FPR, old → new |
| --- | ---: | ---: | ---: | ---: | ---: |
| gloom_exfiltration (2066) | 0.763290 → 0.825344 | 0.918684 → 0.943579 | 0.112373 → 0.106914 | 0.789525 → 0.747818 | 0.078261 → 0.033816 |
| test_stride (946) | 0.929255 → 0.946835 | 0.984104 → 0.988275 | 0.066059 → 0.066073 | 0.871750 → 0.873484 | 0.024390 → 0.010840 |

Ten-bin expected calibration error worsens despite the slightly better Brier
score. The candidate underpredicts positive probability; no calibration is fitted
on this ID set.

| Population | ECE, old | ECE, new | Candidate mean probability minus prevalence |
| --- | ---: | ---: | ---: |
| pooled | 0.058943 | 0.096407 | -0.096407 |
| gloom_exfiltration | 0.051820 | 0.097813 | -0.097813 |
| test_stride | 0.074498 | 0.093336 | -0.093336 |

The score-tie diagnostic records 1,472 unique scores/1,540 `tied_rows`, versus
1,569/1,443 historically. The final prediction SHA256 is
`12e75f3db89e0e811d1236f2bc9337ae2d8397cb1d688fad42520fe7106e08e4`.
Local predictions, summary and both collected adapter files match their recorded
checksums. `summary.json` reports complete, with no promotion and no OOD
evaluation.

ID scoring takes 665.178 seconds, versus the historical 658.275 seconds; these
exclude engine startup and prompt preparation. The successful pipeline takes
5154.132 seconds through its final summary, including training, loading,
reference, serving preparation and scoring. Earlier failed attempts are
additional preserved costs, excluded from that successful-pipeline figure. The
46-minute training loop should not be described as the entire campaign time.

Worker PID 67332 remains alive and idle on pod `i243nsg10usytq`; the evaluation
engine exits normally and GPU memory returns to about 15 GB. Keep its native
plans, packed weights, initial FP32 adapter copies and shared disk caches for
future authorized optimization.

## Worker shutdown and interpretation, 2026-10-06

The user ends training optimization and requests stopping the resident worker.
Its native shutdown request completes cleanly: PID 67332 is absent from `/proc`,
`worker.json` records `stopped`, and GPU memory returns to 0 MiB. Preserve
`worker_shutdown.json`, completed adapters, logs and shared disk caches. This
process shutdown does not terminate the B200 pod; capacity disposition requires
the separate user response.

Higher training loss with improved held-out ranking is consistent with a
regularization hypothesis, but does not identify it. FP4 rounding of frozen MLP
weights and dynamic activations changes both the forward computation and the
input gradients seen by FP32 LoRA adapters. This could impede fitting particular
training examples. Related quantization-aware training research reports implicit
regularization ([QT-DoG, ICML 2025](https://proceedings.mlr.press/v267/javed25a.html)),
but that evidence does not establish the mechanism for this frozen-base LoRA
recipe. The attention backend also changes, only one seed is observed, and ECE
worsens. Do not claim reduced overfitting, flatter minima or causal FP4 gains
without a matched precision-only comparison. No further experiment is launched.
