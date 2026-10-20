# Three-seed weight average for the Gleipnir 4B release

Gleipnir 4B trains on the strict-filtered tool-trajectory data only
([decision](../../docs/decisions.md), 2026-10-20). Its three BF16/SDPA seeds
differ by up to 3.6 OOD and 1.8 ID points from data order alone
([seed variation](../teacher_seed_variation/README.md)). Choosing the
OOD-best seed would select on OOD and report an inflated score. This
experiment instead removes the seed noise by averaging the weights.

- **Hypothesis.** The exact weight average of seeds 0, 1 and 2 ranks held-out
  ID at least as well as the seeds' mean.
- **Intervention.** None during training; there is no training. The seeds
  share initialization, data, prompt and schedule. Their FP32 LoRA updates
  are averaged exactly, as one rank-384 LoRA: each seed's A matrices are
  stacked, its B matrices are divided by 3, and alpha is 768, so the scaling
  stays 2 (`average.py`). The build checks every module's update against the
  explicit average in FP64.
- **Pipeline.** The campaign is the seed-1 control's config with a new
  identity. Its "train" stage is replaced by the build, and everything after
  is the shared path: FP32-master reference, FP32 merge to BF16,
  master/merged parity and ID scoring. A separate OOD context re-merges, must
  reproduce the merge exactly, and scores the 6,395-row OOD suite.
- **Release rule** (frozen before scoring, `compare.py`). The average is the
  release candidate if its ID macro pAUROC@20 is at least the three-seed mean
  (0.9026). Otherwise the ID-best seed is (seed 2). OOD is scored once,
  reported, and never used for selection.
- **Baselines.** Seed 0 rescored (`teacher-seed-sdpa-s0-rescore01`), seed 1
  and seed 2, all under the same BF16 serving.

```bash
PYTHONPATH=src:. python -m experiments.seed_average.prepare --gpu-uuid UUID
PYTHONPATH=src:. <training-python> -m gleipnir.campaigns.monitoring prepare --config data/teacher-seed-sdpa-avg012-01/config.yaml
PYTHONPATH=src:. <training-python> -m experiments.seed_average.run
python -m experiments.seed_average.compare
```

## Result (2026-10-20)

**Release decision under the frozen rule: the average.** Its ID macro
pAUROC@20 is 0.9124, above the three-seed mean of 0.9026
(`results/seed_average/comparison.json`).

| Model | ID macro pAUROC@20 | ID Gloom | ID STRIDE | ID Brier | OOD macro pAUROC@20 | OOD AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Seed 0 | 0.8959 | 0.8285 | 0.9633 | 0.0889 | 0.7869 | 0.9222 |
| Seed 1 | 0.8978 | 0.8168 | 0.9788 | 0.0722 | 0.8125 | 0.9255 |
| Seed 2 | 0.9142 | 0.8619 | 0.9665 | 0.0607 | 0.7768 | 0.9061 |
| Seed mean | 0.9026 | | | | 0.7921 | |
| **Average** | **0.9124** | 0.8517 | 0.9732 | 0.0622 | **0.8108** | 0.9199 |

OOD pAUROC@20 by source:

| Source | Seed 0 | Seed 1 | Seed 2 | Average |
|---|---:|---:|---:|---:|
| AgentDojo | 0.9298 | 0.9248 | 0.8506 | **0.9430** |
| CUA-SHADE | 0.7717 | 0.7640 | 0.7679 | **0.7916** |
| IAC | 0.8123 | 0.8958 | 0.8451 | 0.8845 |
| MLE sabotage | 1.0000 | 1.0000 | 0.9995 | 1.0000 |
| Rogue eval | 0.9696 | 0.9812 | 0.9765 | 0.9804 |
| SHADE-Arena | 0.2380 | 0.3092 | 0.2209 | 0.2653 |

- **ID.** The average is 1.0 point above the seed mean and within 0.2 of the
  best seed. Its Brier (0.0622) is the second best.
- **OOD.** Reported once, never selected on. The average is 1.9 points above
  the seed mean, within 0.2 of the best seed, and 3.4 above the ID-best seed
  (seed 2). It is the best of the four on AgentDojo and CUA-SHADE. It never
  falls below the middle seed on any source.
- **Score shifts.** Versus seeds 0, 1 and 2, mean absolute score differences
  are 0.081, 0.054 and 0.041 on ID, and 0.050, 0.049 and 0.042 on OOD.
- **Pipeline.** The build was exact (maximum relative update error 2.2e-8).
  Merged parity: mean difference 0.0022, correlation 0.99992, maximum
  difference 0.025. The OOD re-merge reproduced the first merge's hashes.
- **Hardware qualification.** The average was merged and scored on an
  EU-NL-1 B300 (`GPU-02557ba6…`, sm_103, driver 595.91.07); the seed
  baselines were scored on B200s. Seed 1 was rescored on ID on the same B300
  (`teacher-seed-sdpa-s1-b300-rescore01`, identical re-merge). It scored
  0.8984, against 0.8978 on the B200: +0.06 points, mean absolute score
  difference 0.0019, 5 of 3,012 threshold flips. The average leads seed 1 on
  the same hardware by 1.40 points.
- **First attempt.** It stopped at the build's own check, because PEFT lists
  target modules in arbitrary order. Its receipts are preserved under
  `*_failed_target_order01`; the fix compares the modules as sets.
- **Evaluation design.** OOD reuses the seed-variation scoring context,
  which merges a second time. One context scoring both splits would avoid
  that.
