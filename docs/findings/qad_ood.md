# Full QAD checkpoint OOD follow-up

The user authorized evaluating the original full QAD checkpoint on the frozen
OOD suite on 2026-10-13, then explicitly requested optimized serving as well.
The [fixed evaluation contract](../../experiments/qad_ood/README.md) runs one
BF16 pass followed by one optimized pass on all 6,395 unchanged rows. This
separates changed checkpoint behavior from deployment precision effects; it
does not retrain, select checkpoints, tune OOD or promote a model.

The fixed FP32 master is the original 245-update same-model QAD result,
SHA256 `eb08e1371e72cde91243b4f0792270aa52ea97e855d9a9099a4b436f85659770`.
Compare the cached original FP4/FA4 BF16 scores and filtered SDPA, released
Gleipnir 4B and unfiltered regular references. The previous FP4/FA4 run has mean
source-macro pAUROC@20 0.737928 versus filtered SDPA 0.786936. Higher-level
lineage independence is unestablished and the OOD suite has been used repeatedly.

The initial driver PID 12524 launched on the verified idle existing NC2 B200.
Artifacts live
under `results/qad-ood01/`, with separate `bf16/` and `optimized/` receipts;
logs are under `logs/runpod/qad-ood01/`. Startup checks bind the source/config,
master/export, reference scores and canary rendering, replacing only the owned
ephemeral merge after predecessor verification. Failed gates stop the queue.
The amended run completed both serving conditions. Capacity remains running;
no in-chat scheduling tool is available.

Seven focused tests pass, covering merge ownership/reuse, GPU launch exclusion,
sequential serving order with separate receipts and stopping after a failed
first condition and leaving variability unmeasured for one pass. Ruff and diff
whitespace checks pass.

The user subsequently reduced both serving paths to one pass. The initial
controller and scorer were stopped early in BF16 scoring; all partial outputs,
source/contract and startup receipts remain under `attempt01/`. The amended
contract retains the fixed checkpoint and inputs, reuses its verified merge and
restarts scoring with fresh canaries as driver PID 12880. No repeat variation
is reported. The amended launch reuses the verified identical QAD merge;
all three fresh BF16 serving canaries and its native audit pass. Both serving conditions pass three fresh canaries and their dispatch checks.
All 6,395 predictions per condition were collected locally, and complete
ranking/calibration metrics were independently recomputed exactly. The scorer
retired successfully; no GPU compute process remains at the completion check.


## Completed comparison

All ranking values below are equal-source macro metrics. The preceding original
FP4/FA4 values average both cached BF16 passes; QAD has one pass per serving path,
per user instruction.

| Checkpoint / serving | pAUROC@20 | AUROC | Prompt tokens/s |
| --- | ---: | ---: | ---: |
| Original FP4/FA4, BF16 | 0.737928 | 0.899799 | ~111k |
| Full QAD, BF16 | 0.760482 | 0.910175 | 111,456 |
| Full QAD, optimized | 0.713660 | 0.890313 | 171,867 |
| Cached strict-filtered SDPA, BF16 | 0.786936 | 0.921876 | historical control |
| Cached released Gleipnir 4B | 0.782350 | 0.908584 | historical control |

QAD improves BF16 pAUROC by 2.2554 percentage points over its original checkpoint,
but remains 2.6453 points below the strict-filtered SDPA control. Optimized serving
loses 4.6822 pAUROC points and 1.9862 AUROC points against the same QAD checkpoint
in BF16. It is 1.5420 times faster (496.58 versus 322.03 seconds for 55,346,879
prompt tokens). Paired serving score MAE is 0.060960, correlation 0.960988, maximum
absolute difference 0.629253, and 376/6,395 threshold-0.5 decisions change.
These are differences between precision paths, not repeat-variability estimates.

| Source | Original FP4/FA4 BF16 | QAD BF16 | QAD optimized |
| --- | ---: | ---: | ---: |
| AgentDojo | 0.891788 | 0.924011 | 0.838543 |
| CUA SHADE | 0.735926 | 0.760025 | 0.744279 |
| IAC | 0.730427 | 0.764356 | 0.695327 |
| MLE sabotage | 1.000000 | 1.000000 | 1.000000 |
| Rogue eval | 0.891374 | 0.916390 | 0.836369 |
| SHADE arena | 0.178057 | 0.198113 | 0.167443 |

BF16 checkpoint improvement spans five sources. The optimized serving penalty
is also broad, largest on AgentDojo, Rogue eval and IAC. Pooled calibration and
operating-point diagnostics likewise worsen: BF16 versus optimized Brier
0.153252 versus 0.184627, ECE 0.133798 versus 0.169827, threshold-0.5 recall
0.652215 versus 0.571429 and FPR 0.064911 versus 0.056444. Both have coarse BF16
score resolution (78 versus 81 pooled unique scores), so retain ties in ranking.

The completed ID recovery therefore does not establish OOD deployment parity.
Passing small training-source canaries fails to protect held-out OOD ranking in
this comparison. The same-checkpoint serving contrast is direct, but has only
one pass per path and does not measure optimized repeat noise. The QAD-versus-
original checkpoint comparison also changes training exposure/objective and
retains historical runtime/host confounds. No OOD promotion, tuning or automatic
recipe change is made. Frozen missing-prose inputs remain unchanged.

Completed artifacts and `verified_comparison.json` live under
`results/qad-ood01/`; interrupted-attempt evidence remains under `attempt01/`.
