# Evaluation experiments

Held-out, OOD and external-benchmark evaluation, audits and base-model screens. Add new experiments to this table.

| Experiment | Question |
| --- | --- |
| [`aisi_lie_detection_transfer`](aisi_lie_detection_transfer/) | AISI lie-detection external transfer |
| [`cascade`](cascade/) | Gleipnir 4B → 27B cascade |
| [`decision_models_ood`](decision_models_ood/) | Decision models on the strict-OOD suite |
| [`flint_ood`](flint_ood/) | Flint on the strict OOD suite |
| [`gleipnir4b_full_specification`](gleipnir4b_full_specification/) | Gleipnir 4B scored with the full specification |
| [`id_action_only_evaluation`](id_action_only_evaluation/) | Gleipnir 4B cleaned-ID evaluation |
| [`id_assistant_prose_audit`](id_assistant_prose_audit/) | ID assistant-prose removal audit |
| [`id_cot_only_evaluation`](id_cot_only_evaluation/) | ID evaluation with only identified CoT removed |
| [`inkling_ood`](inkling_ood/) | Inkling 2 Mini on the strict OOD suite |
| [`monitor_score_resolution`](monitor_score_resolution/) | Final two-logit score resolution |
| [`monitoring_length_shortcut_audit`](monitoring_length_shortcut_audit/) | Monitoring length-shortcut audit |
| [`ood_reasoning_audit`](ood_reasoning_audit/) | Frozen OOD reasoning audit |
| [`ood_recall_at_fpr`](ood_recall_at_fpr/) | Recall at 1% FPR on the OOD suite |
| [`ood_visible_prose`](ood_visible_prose/) | Restore the original public monitoring surface |
| [`openai_monitoring_ood`](openai_monitoring_ood/) | OpenAI full-teacher OOD monitoring benchmark |
| [`pplx_decider_ood`](pplx_decider_ood/) | pplx-decider v1.2 on the strict-OOD frontier |
| [`pretrained_id`](pretrained_id/) | Pretrained-only Qwen3.5 ID comparison |
| [`qwen27b_generations_ood`](qwen27b_generations_ood/) | Qwen 27B generations on the strict OOD suite |
| [`shade_length_audit`](shade_length_audit/) | SHADE-Arena length audit |
| [`sleightbench_eval`](sleightbench_eval/) | SleightBench single-pass evaluation of the released monitors |
| [`tool_output_ablation`](tool_output_ablation/) | Monitoring without tool outputs |
