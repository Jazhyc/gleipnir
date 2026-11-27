# Training experiments

Monitor training recipes, objectives, data and capacity scaling, releases. Add new experiments to this table.

| Experiment | Question |
| --- | --- |
| [`adapter_capacity_scaling`](adapter_capacity_scaling/) | QLoRA adapter-capacity scaling on full Kimi data |
| [`aux_supervision`](aux_supervision/) | Auxiliary Position/Reason supervision for the 4B monitor |
| [`data_capacity_scaling`](data_capacity_scaling/) | Joint data-volume and QLoRA-rank scaling |
| [`data_scaling`](data_scaling/) | Kimi K3 data scaling |
| [`deception_distillation`](deception_distillation/) | Deception distillation |
| [`distillation_temperature`](distillation_temperature/) | Distillation temperature |
| [`final_training_fit`](final_training_fit/) | Final-checkpoint teacher fit on the full training population |
| [`hard_label_seeds`](hard_label_seeds/) | Hard-label 4B seeds matched to the soft-only release seeds |
| [`monitoring_campaign`](monitoring_campaign/) | Monitoring campaign runner |
| [`monitoring_duration`](monitoring_duration/) | Monitoring-only training duration |
| [`monitoring_hard_labels`](monitoring_hard_labels/) | Tool-trajectory hard-label strength screen |
| [`monitoring_id_scaling`](monitoring_id_scaling/) | Monitoring-only ID data scaling |
| [`monitoring_lr_sweep`](monitoring_lr_sweep/) | Gleipnir 4B monitoring-only learning-rate screen |
| [`monitoring_mil_mixture`](monitoring_mil_mixture/) | Three-epoch MIL with full deception data |
| [`monitoring_mil_ood`](monitoring_mil_ood/) | ID-selected three-epoch MIL: frozen OOD evaluation |
| [`monitoring_objective_ablation`](monitoring_objective_ablation/) | Monitoring objective ablation |
| [`monitoring_prefix_supervision`](monitoring_prefix_supervision/) | Intermediate tool-action supervision |
| [`monitoring_subset_duration`](monitoring_subset_duration/) | Duration, MIL and sampled prefixes on the frozen 20% subset |
| [`optimization_regularization_screen`](optimization_regularization_screen/) | Optimization and regularization screen |
| [`optimizer_full_data_confirmation`](optimizer_full_data_confirmation/) | Full-data confirmation of RMS-matched Muon |
| [`optimizer_lr_scaling`](optimizer_lr_scaling/) | Scheduled AdamW versus RMS-matched Muon |
| [`partial_auroc`](partial_auroc/) | One strict-filtered partial-AUROC variant |
| [`qwen38_27b_campaign`](qwen38_27b_campaign/) | First Qwen3.8-27B monitor campaign |
| [`qwen38_27b_hard_labels`](qwen38_27b_hard_labels/) | Hard-label Qwen3.8-27B matched to the soft 27B seed 0 |
| [`reasoning_sft_scaling`](reasoning_sft_scaling/) | Corrected reasoning-SFT data scaling |
| [`seed_average`](seed_average/) | Three-seed weight average for the Gleipnir 4B release |
| [`tool_trajectory_monitoring`](tool_trajectory_monitoring/) | Binary action-only tool-trajectory monitoring |
| [`training_procedure_screen`](training_procedure_screen/) | Structural training-procedure screen |
