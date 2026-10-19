# Gleipnir 27B as the 4B student's teacher

**Hypothesis.** Gleipnir 27B scored under the full teacher prompt is a better
teacher for the Qwen3.5-4B student than Kimi K3. On OOD it reaches 0.9352
Mean-OOD pAUROC@20 against Kimi K3's 0.9084
([vocab distillation](../gleipnir27b_vocab_distillation/README.md)). If it is
the better teacher, distilling its binary 0/1 targets raises held-out ID macro
pAUROC@20 above the seed-matched Kimi-distilled control. A second arm asks
whether the 4B also gains from reading the teacher prompt itself.

**Caveat on the teacher.** Gleipnir 27B was trained on Kimi's soft targets
for 7,836 of these 8,688 pool rows, so on those rows its targets are a refit
of Kimi (training-row margin vs. Kimi logit correlation 0.983). Its own
judgment shows mainly through the teacher prompt and on the 852 rows Kimi's
strict filter removed.

## Design (frozen before training)

**Arms.** Both copy the seed-1 control `teacher-seed-sdpa-s1-01`: BF16/SDPA,
untrimmed `qwen35_4b_b200_packed_sdpa` as in the control, LoRA 128, LR 5e-5,
one epoch, soft BCE only. Both arms change the teacher; the second also
changes the prompt.

| arm | campaign | 4B input (training and ID) |
|---|---|---|
| student prompt | `gleipnir27b-teacher-student-s1-01` | compact student prompt |
| teacher prompt | `gleipnir27b-teacher-teacherprompt-s1-01` | full teacher prompt |

- **Targets.** `prepare.py` reads Gleipnir 27B's teacher-prompt extraction
  (`extract01/gleipnir_teacher_train`: stock vLLM BF16, FP32 LM head) for all
  8,688 pool rows: `soft_target = sigmoid(logit_1 − logit_0)`, no
  temperature. Rows join by id, and every raw and rendered prompt hash, label
  and source is checked.
- **Filter.** The recipe's strict filter (delta 0) runs on the new teacher.
  Expected from the extraction: it keeps 8,249 rows and removes 439, against
  Kimi's 7,836 (7,800 rows shared). So both teacher and row set change.
- **Context cap.** The teacher prompt adds about 800 tokens per row; 10
  retained rows exceed the profile's 29,696-token cap under it (27B count).
  They are removed from both arms, without truncation, so the two arms train
  on identical rows and targets (expected 8,239).
- **Teacher-prompt rendering.** Training rows are re-rendered with the
  teacher template (`rerender_training`). This reproduces all 8,688 prompts
  Kimi and Gleipnir scored byte for byte. ID uses the canonical CoT-removed
  ID trajectories rendered with the teacher template, not the frozen teacher
  ID prompts. Those keep STRIDE agent reasoning and `think` calls on 937
  rows, which the [CoT-removed validation decision](../../docs/decisions/cot_removed_id_validation.md)
  excludes.

**Baselines.** Seed-1 Kimi control (0.8978 ID macro pAUROC@20); Kimi seeds 0
and 2 (0.8959, 0.9142) as seed context; the unadapted-Qwen3.8 teacher arm
(0.8792) as the earlier teacher swap.

**Selection.** The final checkpoint of each arm, scored on ID only. The
frozen advancement rule from `distillation_temperature/compare.py`: macro
gain ≥ 1.0 point, no source loss > 1.0 point, Brier rise ≤ 0.005. Advancing
means more seeds, never promotion. The teacher-prompt arm is compared on the
same 3,012 ID rows, but under its own prompt.

**Stop** on input/runtime drift, nonfinite or missing gradients, failed
master/merge/serving parity, incomplete scoring, or completion. Each arm
releases its merged slot after scoring.

**Teacher reference on canonical ID.** Before training, Gleipnir 27B is
rescored under the teacher prompt on the CoT-removed ID trajectories
(`extract.py --jobs teacher:id_cot_removed`). This gives the teacher's own
ID quality on the inputs the students are scored on.

```bash
PYTHONPATH=src:. python -m experiments.gleipnir27b_teacher_4b.prepare --gpu-uuid UUID
PYTHONPATH=src:. <training-python> -m gleipnir.campaigns.monitoring prepare --config data/<arm>/config.yaml
PYTHONPATH=src:. <training-python> -m experiments.gleipnir27b_teacher_4b.run
python -m experiments.gleipnir27b_teacher_4b.compare
```

## Result (2026-10-18 23:30 – 2026-10-19 01:48 UTC, B200 `GPU-4b31de30…`)

Both arms completed every stage: train, master reference, merge, merged
parity (correlation 0.99996 student, 0.99988 teacher prompt) and full ID
scoring (3,012 rows). The student-prompt arm took 67 min, the teacher-prompt
arm 71 min. Both merged slots were released. Preparation: 439 rows removed by
the strict filter, 10 over the cap, 8,239 trained in both arms.

| ID | Kimi seed 1 (control) | Kimi seed 0 | Kimi seed 2 | **Gleipnir, student prompt** | **Gleipnir, teacher prompt** |
|---|---:|---:|---:|---:|---:|
| macro pAUROC@20 | 0.8978 | 0.8959 | 0.9142 | **0.8773** | **0.8762** |
| macro AUROC | 0.9627 | 0.9638 | 0.9734 | 0.9545 | 0.9600 |
| macro Brier | 0.0722 | 0.0889 | 0.0607 | 0.0810 | 0.0876 |
| Gloom exfiltration pAUROC@20 | 0.8168 | 0.8285 | 0.8619 | 0.7924 | 0.7840 |
| Test STRIDE pAUROC@20 | 0.9788 | 0.9633 | 0.9665 | 0.9621 | 0.9684 |
| macro FPR at 0.5 | 0.121 | | | 0.130 | 0.183 |

**Decision: neither arm advances.** Student prompt: −2.05 macro points,
Gloom −2.44, STRIDE −1.66, Brier +0.0088. Teacher prompt: −2.16 macro,
Gloom −3.28, STRIDE −1.04, Brier +0.0154. Each fails all three conditions.
Scores against the control: MAE 0.080 / 0.086, 271 / 310 rows crossing 0.5.

**Reading.**
- Both Gleipnir-distilled students fall below all three Kimi seeds, at the
  level of the unadapted-Qwen3.8 teacher swap (0.8792), even though this
  teacher is far stronger than Kimi on OOD (0.9352 vs 0.9084) and its own
  canonical ID is 0.9420.
- The teacher prompt gives the 4B no ranking gain on pAUROC@20 (−0.1 point
  against the student prompt; AUROC +0.6). It shifts scores upward instead:
  recall rises, FPR at 0.5 goes from 0.130 to 0.183, and Brier worsens. The
  27B's +2.0-point teacher-prompt gain does not transfer at 4B scale.
- Since the targets refit Kimi on most rows (correlation 0.983), the loss
  likely comes from where they differ: the changed row set (8,239 vs 7,836;
  the strict filter now keeps rows Kimi's removed) and the sharper
  FP32-head targets. This run cannot separate the two.
- Limits: one seed per arm. The Kimi seed spread is 1.8 points, about the
  size of these gaps; Gloom in both arms is below the Kimi seed range
  (0.817–0.862).
