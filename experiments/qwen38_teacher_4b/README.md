# Qwen3.8-27B as the 4B student's teacher

**Hypothesis.** A same-family teacher shares more transferable "dark
knowledge" with the Qwen3.5-4B student than Kimi K3 does. If so, distilling
from unadapted Qwen3.8-27B raises held-out ID macro pAUROC@20 above the
seed-matched Kimi-distilled control, even though Qwen3.8 is the weaker
teacher (training rows: 0.930 vs. 0.964 macro pAUROC@20;
[comparison](../qwen27b_generations_ood/README.md)).

## Design (frozen before training)

**Arm.** `qwen38-teacher-s1-01` copies the seed-1 control
`teacher-seed-sdpa-s1-01` (BF16/SDPA, LoRA 128, LR 5e-5, one epoch, soft
BCE only). Only the teacher changes:
- **Targets.** `prepare.py` writes Qwen3.8-27B binary targets for all 8,688
  pool rows from `results/qwen27b_generations_ood/qwen38_27b_train`: full
  teacher prompt, non-thinking boundary, `soft_target = sigmoid(logprob_1 −
  logprob_0)`. Rows join by id, with prompt hash, label and source checked.
- **Filter.** The recipe's strict filter (delta 0) runs on the new teacher. It
  keeps 7,874 rows and removes 814; 571 of those are also removed by Kimi's
  filter. So both teacher and row set change. That is the recipe applied to a
  different teacher, not a row-matched swap.
- **Profile.** The untrimmed `qwen35_4b_b200_packed_sdpa`, as in the control,
  so the trim is not a second change.

**Baselines.** Seed-1 Kimi control (0.8978 ID macro pAUROC@20); seeds 0 and 2
(0.8959, 0.9142) as seed context.

**Selection.** The final checkpoint, scored on ID only. The frozen advancement
rule from `distillation_temperature/compare.py`: macro gain ≥ 1.0 point, no
source loss > 1.0 point, Brier rise ≤ 0.005. Advancing means more seeds, never
promotion.

**Stop** on input/runtime drift, nonfinite or missing gradients, failed
master/merge/serving parity, incomplete scoring, or completion. After scoring,
the run releases its merged slot.

```bash
PYTHONPATH=src:. python -m experiments.qwen38_teacher_4b.prepare --gpu-uuid UUID
PYTHONPATH=src:. <training-python> -m gleipnir.campaigns.monitoring prepare --config data/qwen38-teacher-s1-01/config.yaml
PYTHONPATH=src:. <training-python> -m experiments.qwen38_teacher_4b.run
python -m experiments.qwen38_teacher_4b.compare
```
