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

## Result (2026-10-17 01:54–02:57 UTC, B200 `GPU-3d8d6bde…`)

All stages completed: train, master reference, merge, merged parity
(correlation 0.99986) and full ID scoring (3,012 rows). The driver took
62 minutes end to end. The merged slot was released.

| ID | Kimi seed 1 (control) | Kimi seed 0 | Kimi seed 2 | **Qwen3.8 teacher, seed 1** |
|---|---:|---:|---:|---:|
| macro pAUROC@20 | 0.8978 | 0.8959 | 0.9142 | **0.8792** |
| macro AUROC | 0.9627 | 0.9638 | 0.9734 | 0.9589 |
| macro Brier | 0.0722 | 0.0889 | 0.0607 | 0.0716 |
| Gloom exfiltration pAUROC@20 | 0.8168 | 0.8285 | 0.8619 | 0.7770 |
| Test STRIDE pAUROC@20 | 0.9788 | 0.9633 | 0.9665 | 0.9815 |

**Decision: does not advance.** Macro pAUROC@20 is −1.86 points against the
control, and Gloom is −3.98 (the rule allows −1.0). STRIDE is +0.27; Brier is
flat. Scores move a lot relative to the control: MAE 0.103, with 286 rows
crossing 0.5.

**Reading.**
- The Qwen3.8-distilled student is below all three Kimi-distilled seeds,
  and below its own unadapted teacher on ID (0.8869).
- At this scale, same-family transfer does not make up for the weaker
  teacher. The loss is concentrated on Gloom, where unadapted Qwen3.8 is
  itself strong (0.845), so the student did not inherit it.
- Limits: one seed, and the row set changed with the teacher (7,874 vs.
  7,836 rows). The Kimi seed spread on macro pAUROC@20 is 1.8 points, about
  the size of this gap, but Gloom's −4.0 falls outside the Kimi seeds'
  0.817–0.862 range.
