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
