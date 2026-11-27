# Teachers experiments

Teacher selection, agreement, filtering and teacher-student distillation sources. Add new experiments to this table.

| Experiment | Question |
| --- | --- |
| [`filter_margin_ood`](filter_margin_ood/) | OOD scores for the teacher-agreement filter margins |
| [`filtered_fp4_baseline`](filtered_fp4_baseline/) | Strict-filtered FP4/FA4 baseline |
| [`filtered_fp4_ood`](filtered_fp4_ood/) | Strict-filtered FP4 MLP/FA4 OOD evaluation |
| [`filtered_hard_labels`](filtered_hard_labels/) | Hard labels on the strict-filtered recipe |
| [`gleipnir27b_teacher_4b`](gleipnir27b_teacher_4b/) | Gleipnir 27B as the 4B student's teacher |
| [`gleipnir27b_vocab_distillation`](gleipnir27b_vocab_distillation/) | Gleipnir 27B as a full-vocabulary teacher for the 4B |
| [`qwen122b_id`](qwen122b_id/) | Qwen3.5-122B-A10B FP8 ID teacher screen |
| [`qwen38_teacher_4b`](qwen38_teacher_4b/) | Qwen3.8-27B as the 4B student's teacher |
| [`teacher_agreement`](teacher_agreement/) | Matched Qwen/Kimi full-trajectory agreement |
| [`teacher_disagreement_filtering`](teacher_disagreement_filtering/) | Teacher-disagreement filtering sweep |
| [`teacher_disagreement_ood`](teacher_disagreement_ood/) | Strict-filtering OOD follow-up |
| [`teacher_precision_factorial`](teacher_precision_factorial/) | Teacher precision factorial (MLP precision × attention) |
| [`teacher_seed_variation`](teacher_seed_variation/) | Seed variation of the strict-filtered SDPA teacher |
