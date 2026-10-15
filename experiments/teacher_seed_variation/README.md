# Seed variation of the strict-filtered SDPA teacher

First step of the teacher-precision study. The goal is to separate the MLP
(FP4 vs BF16) and attention (FA4 vs SDPA) contributions to the FP4/FA4
teacher's OOD loss. Only the two diagonal cells exist, each with one seed.
Before attributing a difference between cells, measure how much one teacher
varies across seeds.

**Hypothesis.** Retraining the strict-filtered BF16/SDPA teacher
(`teacher-disagreement-delta000-sdpa04`, seed 0) with seed 1 changes OOD and
ID ranking much less than the FP4/FA4 gap (OOD −4.90 pAUROC@20 points, ID
−0.89).

**Intervention.** `teacher-seed-sdpa-s1-01` copies the seed-0 campaign config
and changes only the seed, identity, GPU, merge slot and contract text. The
seed controls data order and packing. Initialization (`a6b1d2e9…`), data,
Kimi targets, profile, schedule and packing budget are unchanged; a test
enforces this. The serving-selection hash is updated because the serving
default changed after seed 0. BF16 scoring replaces the optimized workers.

**Scoring.** All runs use BF16 vLLM 0.31 at c128. Seed 1 is scored on ID
(3,012 rows) and the frozen OOD suite (6,395 rows). Seed 0 is rescored on
both in the same serving session, as a drift control against its cached
scores.

**Selection.** Descriptive only; no checkpoint is promoted. Under the
[precision-selection decision](../../docs/decisions/precision_selection_on_ood.md),
later precision cells may select on OOD. This run sets the noise floor for
that.

**Runtime.** Training runs on the EU-RO-1 B200 with the NC2 training runtime
restored from `container_runtimes.tar.zst` (SHA-256 `c4d76a28…`). The original
seed-0 run used a different host and GPU, so seed and host effects are
confounded at the seed-0/seed-1 boundary. Same-host repeatability of this
recipe is bitwise for BF16/FA4 (augmented line) and assumed, not measured,
for SDPA.

**Stop** on input or runtime drift, nonfinite or missing gradients, failed
master/merge/serving parity, incomplete scoring, or completion.

```bash
PYTHONPATH=src:. <training-python> -m experiments.teacher_seed_variation.prepare --gpu-uuid UUID
PYTHONPATH=src:. <training-python> -m gleipnir.campaigns.monitoring prepare \
  --config data/teacher-seed-sdpa-s1-01/config.yaml
PYTHONPATH=src:. <training-python> -m experiments.teacher_seed_variation.run
python -m experiments.teacher_seed_variation.compare
```
