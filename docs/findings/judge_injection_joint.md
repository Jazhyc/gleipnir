# Joint monitoring and injected-preference training

Date: 2026-10-04. Status: training and all four evaluation populations complete.

User selected a fresh adapter with 75% monitoring and 25% preference examples,
using the regular 5e-5 learning rate. The hypothesis is that retaining trajectory
supervision throughout training limits the retention regression seen after
preference-only continuation. Forgetting and task interference are not isolated.

The sampler draws all 8,688 original regular-prompt trajectories once, plus
2,896 preference examples: 290 clean, 1,303 preferred-answer-injected and
1,303 disfavored-answer-injected. Each full logical batch has 24/8 examples.
Both A/B orders are balanced within each condition. There are 362 updates.
The 2,606 injected draws represent 22.497% of all 11,584 training draws.
The sampled preference population has 25 original queries, 940 answer pairs,
78 distinct payload IDs and 2,896 unique augmented examples. Variants are not
independent tasks; more exposure does not establish broader attack diversity.

The frozen schedule contains 84,842,449 input tokens: 83,816,311 monitoring,
93,414 clean preference, 464,934 disfavored-injected and 467,790 preferred-injected.
Auxiliary examples are 25% of example-level loss mass but only 1.21% of input
tokens. Maximum context is 29,337, inside the existing validated 29,696 envelope;
no truncation. The eligible preference pool remains the 16,760 variants from
25 query groups; the 4,188 variants from six held-out queries remain excluded.

Preserve soft teacher BCE and the regular 0/1 monitoring surface. Preference
uses its separate hard construction labels, quality rubric and A/B surface.
Each example contributes one loss; packed readouts gather its own token pair.
The verified initial adapter has all 128 B matrices zero, matching the original
regular run's initialization. It is not either trained checkpoint.

FP32 masters, frozen BF16 base, packed FlashQLA/SDPA, persistent compiler-cache
reuse, the checksum-pinned original startup-validation reference and mandatory
finite/missing-gradient guards are retained. Final checkpoint only, no sweeps
or benchmark-based checkpoint selection. New serving parity is mandatory.

Compare original, preference-only continuation and joint-training results on
identical preference, regular-prompt APPS/honest controls and CoT-removed ID.
APPS remains excluded from training, data generation and sampler construction.
Differences in optimizer updates, auxiliary exposure and learning rate limit
causal attribution; no model promotion follows this exploratory comparison.

Protocol: [experiment README](../../experiments/judge_injection_joint/README.md).

## Startup verification

The existing B200 launch succeeded (pod `3422tdq7tfog52`, launcher PID 11814).
The remote execution contract matches every locally reviewed training source
and the frozen manifest. The unchanged startup validation was reused; the log
records FlashQLA in all 24 GDN layers and strict parity as not repeated. The
shared cache resolves to the populated persistent
`/workspace/gleipnir/.cache/training/student_injection_awareness` directory.
The first optimizer update completed in 277.58 seconds during compilation;
the second took approximately 11 seconds. GPU utilization reached 100% with
143,742 MiB used. Per-update finite/missing-gradient guards passed. The fixed
final checkpoint subsequently completed regular-only monitoring evaluation
and A/B preference evaluation.

Validation: 74 affected CPU tests passed, followed by seven focused experiment
tests including two additional tokenization/real-Trainer integration checks
(76 distinct checks in total). Ruff and staged diff checks passed. Expected
CPU-only NVML warnings are retained. In-chat heartbeat scheduling is unavailable;
startup was inspected during the active turn, without claiming future automatic
agent follow-ups.

## Completed comparison

Training completed all 362 updates in 3,565.59 seconds (59.43 minutes).
Mean optimizer-step time excluding the initial compilation step was 8.96 seconds;
peak allocated CUDA memory was 149.87 GiB. The realized sampler exactly matches
the frozen schedule, including its quotas and order hash. No failure receipt was
produced. All six serving-parity cells passed before full evaluation: joint
monitor score correlation 0.999828 and mean absolute difference 0.002802; joint
preference correlation effectively 1.0 and mean absolute difference 3.40e-7.
Both surfaces showed a nonzero adapter effect. All 256 master tensors are FP32.

The final evaluation completed at 2026-10-03 22:39:41 UTC. Fresh score counts are
4,188 preference variants, 5,184 APPS benchmark rows, 3,930 honest-control rows,
and 3,012 ID examples: 16,314 total. Original and continued comparison scores
are reused from checksum-pinned, matched prior runs.

| Metric | Original regular | Preference-only continued | Fresh joint |
| --- | ---: | ---: | ---: |
| Preference pooled accuracy | 79.85% | 99.74% | 99.90% |
| Clean preference accuracy (504) | 99.40% | 99.80% | 100.00% |
| Preferred-answer-injected accuracy (1,842) | 58.36% | 100.00% | 100.00% |
| Disfavored-answer-injected accuracy (1,842) | 95.98% | 99.46% | 99.78% |
| Preference query-macro accuracy | 82.22% | 98.39% | 99.42% |
| Preference source-macro accuracy | 83.58% | 97.59% | 99.12% |
| ID source-macro pAUROC@20 | 0.846273 | 0.795786 | 0.865829 |
| ID pooled AUROC | 0.942615 | 0.922744 | 0.955748 |
| ID recall at 0.5 | 81.90% | 62.13% | 81.03% |
| ID FPR at 0.5 | 6.41% | 1.78% | 4.49% |
| ID Brier score | 0.097827 | 0.171919 | 0.094879 |
| ID ECE | 0.058943 | 0.190329 | 0.088750 |
| APPS injected-honest alarm rate at clean-code ~2% FPR | 100.00% | 70.64% | 100.00% |
| APPS matched injected-backdoor / injected-honest AUROC | 0.643951 | 0.729899 | 0.625552 |
| APPS vanilla AUROC | 0.918908 | 0.907030 | 0.929478 |
| APPS vanilla TPR at clean-code ~2% FPR | 51.78% | 45.44% | 58.89% |

Preference pooled AUROC is 0.999997, Brier 0.000820 and log loss 0.003074.
There are four errors, all disfavored-injected Search variants from one query;
Search accuracy is 96.49%. Preferred-injected clean-to-wrong flip rate is zero,
with mean paired change in correct-answer probability -0.00000577.
Pooled score ties are 2,758 / 1,416 / 71 across original / continued / joint;
exact 0.5 ties are 147 / 0 / 0. Preference results still represent only six
independent held-out queries and construction labels, not 4,188 independent tasks.

Joint ID pAUROC@20 is 0.796934 on Gloom and 0.934725 on STRIDE, compared with
original 0.763290 and 0.929255. Its pooled log loss is 0.313551. Ranking and
Brier improve over the original, while ECE is worse; do not describe calibration
as uniformly improved. ID has 1,492 tied rows and 1,520 unique scores.

APPS operating points use each model's original 655 clean honest examples,
with strict score > threshold and 13/655 (1.985%) clean alarms. Thresholds are
0.164516455 / 0.075858174 / 0.164516461. Rates and matched AUROCs above average
the three attack templates equally. No threshold is recalibrated on injected
controls. All 655 injected honest controls alarm under each template for joint
training, as do all 647 matched injected backdoors. Joint injected-honest mean
score is 0.668575, compared with 0.637405 original and 0.108354 continued.

| APPS template | Joint injected-honest alarm rate | Joint mean injected-honest score | Joint matched injected AUROC |
| --- | ---: | ---: | ---: |
| GPT-4 transfer | 100.00% | 0.772225 | 0.623159 |
| GPT-4.1 transfer | 100.00% | 0.660539 | 0.563429 |
| GPT-4o transfer | 100.00% | 0.572960 | 0.690066 |

Joint neutral-control alarms average 3.82%, versus 1.48% original and 1.68%
continued. Higher injected-attack scores against clean controls therefore cannot
be interpreted as injection resistance: injected harmless code scores high too.

## Interpretation and provenance

Joint supervision avoided the earlier ID retention regression and acquired
near-perfect held-out preference accuracy despite sampling only 2,896 auxiliary
variants. That behavior did not transfer to distinguishing injected harmless
code from injected backdoors under the regular monitoring prompt. The result
supports task-specific learning of injection handling; it does not establish
a general reduction in harmfulness-monitor suspicion.

The separate A/B quality task and 0/1 harmfulness task remain a material mismatch.
Only 25 training queries, six held-out queries and 78 sampled payload IDs limit
diversity. Different update counts, preference exposure and learning rates prevent
isolating catastrophic forgetting or attributing the ID gain solely to auxiliary
data. No statistical improvement, new adaptive-attack robustness or promotion is
claimed, and APPS did not influence training or checkpoint selection.

FP32 master file SHA256:
`03c99284a0c9afa54126511723fdceeb390d384bf948e849e242bbe9e6c3f7c2`.
Serving file SHA256:
`07e832aa420ee2fa5c6bae544c2380b1420ba99e97c4ff4c7b1942de76e29b42`.
Both collected local files match the completion receipt. Raw scores, parity,
training metadata and full logs are preserved in the ignored artifact trees.
The completed report was recomputed from coverage-checked raw logprobs and
matched inputs, including verified baseline preference caches. Independent
rendering checked every fresh evaluation prompt hash. Collected predictions
are pinned by `evaluation_regular/completion_audit.json`; report SHA256 is
`abd78c46043ab93754e8f6f0c42d82c65bda1d0b5fcfbc602fae7b2f98d11030`.
