# Evaluate Gleipnir 27B under the full teacher prompt

Date: 2026-10-20. Status: recorded in the decision log.

- User decision: by default, Gleipnir 27B is evaluated under the full teacher
  prompt, not the compact student prompt it was trained on. This applies to
  BF16 and optimized/QAD serving comparisons.
- Evidence: under the teacher prompt the same checkpoint reaches 0.9420 CoT-removed
  ID and 0.9352 Mean-OOD pAUROC@20, against 0.9216 and 0.9087 under the
  student prompt. The prompt was chosen on ID
  ([vocab distillation](../../../experiments/teachers/gleipnir27b_vocab_distillation/README.md)).
  The frontier registry already enters the 27B this way.
- Gleipnir 4B keeps its compact student prompt. Optimized-serving gaps for
  the 27B are measured on the teacher-prompt workload
  (`experiments/serving/optimized_gap/teacher_workload.py`).
