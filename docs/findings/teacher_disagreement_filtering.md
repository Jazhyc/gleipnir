# Teacher-disagreement filtering on monitoring training data

The completed one-seed ID sweep finds the strongest ranking under strict
teacher/source-label agreement filtering. Source-macro raw normalized pAUROC@20
increases from **0.846812 to 0.895914 (+4.91 percentage points)**. Keeping a
0.05 uncertainty margin also improves ranking; wider margins do not produce a
monotonic benefit. Strict filtering substantially increases false positives at
the frozen 0.5 threshold. The subsequent
[strict OOD follow-up](teacher_disagreement_ood.md) finds a small, uneven
ranking gain over cached Gleipnir 4B. No model promotion occurred.

## Contract and population

The [experiment contract](../../experiments/teacher_disagreement_filtering/README.md)
freezes original monitoring trajectories and Kimi K3 soft targets, without
synthetic augmentation, deception-mixture data, hard-label loss or target
rewriting. Remove disagreements at `p_harmful >= 0.5` for delta zero. For positive
delta, remove harmful-labelled rows below `0.5-delta` and benign-labelled rows
above `0.5+delta`, preserving equality at either band edge. Retained student and
teacher bytes/order are unchanged. No rule splits an exact-content duplicate
group; original duplicate exposure weights remain.

| Arm | Removed | Remaining | Training tokens | Updates |
|---|---:|---:|---:|---:|
| Unfiltered control, reused | 0 | 8,688 | 83,816,369 | 272 historical |
| Delta 0 | 852 | 7,836 | 75,851,675 | 245 |
| Delta 0.05 | 796 | 7,892 | 76,653,374 | 247 |
| Delta 0.10 | 723 | 7,965 | 77,731,530 | 249 |
| Delta 0.20 | 627 | 8,061 | 78,740,663 | 252 |

All new arms use seed 0, the original zero-B rank-128 FP32 initialization,
one epoch of BF16/SDPA/FlashQLA training, AdamW 5e-5 and logical batch 32.
The existing `student_injection_awareness/4b/regular` control passes the current
completion validator, with matching initialization, data/targets and recipe.
Its training is explicitly reused; historical source/runtime/host differences
and unmeasured training-repeat variability remain comparison qualifications.

Every arm receives fresh FP32-master/BF16-merge/BF16-serving checks and scores
all 3,012 canonical CoT-removed ID examples under the same BF16 vLLM 0.31
recipe. Only final one-epoch checkpoints are evaluated. Filtering changes
source/class composition, exposure and schedule length, so this sweep does not
isolate sample selection from those effects. No random-removal control or
additional seed was requested for this initial sweep.

## ID results

All columns below are unweighted source-macro values. FPR and recall use the
unchanged threshold 0.5; Brier is lower-is-better.

| Arm | pAUROC@20 | AUROC | Brier | FPR | Recall |
|---|---:|---:|---:|---:|---:|
| Control | 0.846812 | 0.951720 | 0.088991 | 5.28% | 82.98% |
| Delta 0 | **0.895914** | **0.963772** | 0.088954 | 21.45% | 94.37% |
| Delta 0.05 | 0.878822 | 0.959360 | **0.075463** | 14.34% | 92.56% |
| Delta 0.10 | 0.836371 | 0.945523 | 0.091658 | 12.66% | 88.07% |
| Delta 0.20 | 0.851275 | 0.945365 | 0.090787 | 4.14% | 82.05% |

| Arm | Gloom pAUROC@20 | STRIDE pAUROC@20 | Gloom FPR at 0.5 | STRIDE FPR at 0.5 |
|---|---:|---:|---:|---:|
| Control | 0.764162 | 0.929462 | 8.12% | 2.44% |
| Delta 0 | **0.828524** | 0.963304 | 38.84% | 4.07% |
| Delta 0.05 | 0.788923 | **0.968722** | 24.35% | 4.34% |
| Delta 0.10 | 0.726653 | 0.946088 | 22.61% | 2.71% |
| Delta 0.20 | 0.743888 | 0.958661 | 6.38% | 1.90% |

Strict filtering improves ranking on both ID sources, with a large Gloom
fixed-threshold false-alarm increase. Its nearly unchanged macro Brier hides
worse Gloom Brier and improved STRIDE Brier. Delta 0.05 has the
best macro Brier and STRIDE ranking. The two wider margins regress Gloom
ranking; delta 0.20's slight macro pAUROC gain coexists with worse macro AUROC.
These observations do not establish a deployable threshold, robust causal
filtering benefit or OOD transfer. Disagreements may involve teacher errors,
source-label errors or rubric/authorization differences.

## Execution and verification

The detached `sweep04` queue completed on 2026-10-11 at **20:51 CEST**, in
5h04m20s. The four fresh training epochs took approximately 49m48s, 50m12s,
50m56s and 51m33s; remaining time includes contract checks, references, merges
and ID scoring. The reused control was evaluated without retraining. Earlier
replaced/failed startup receipts remain preserved separately.

At ID completion the NC2 B200 was idle after retiring the final scorer. All
FP32 adapters remain on the persistent workspace volume. The user-requested
shared ephemeral merge slot retained only the last ID arm's merge; the OOD follow-up later replaced it
with the reconstructed strict arm. Prior manifests/checksums and retirement
receipts remain persistent, with reconstruction required for superseded weights.
No capacity lifecycle action was taken after completion.

Predictions, full source/pooled calibration and ranking diagnostics, configurations,
parity/native receipts and logs are collected locally under
`results/teacher-disagreement-{control,delta000,delta005,delta010,delta020}-sdpa04/`
and `results/teacher_disagreement_filtering/sweep04/`. The independent local
`completion_audit.json` verifies all **15,060 predictions**, canonical ordered
membership, raw score responses, configuration/prediction hashes, complete metric
recomputation and passing BF16/parity gates. The aggregate `comparison.md` and
`summary.json` preserve all arms, including negative results. Source implementation
is commit `7091dee`; 45 focused tests and lint passed before launch.
