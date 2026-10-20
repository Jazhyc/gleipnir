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
