# Injection experiments

Prompt-injection robustness: detection, augmentation, directions, censuses. Add new experiments to this table.

| Experiment | Question |
| --- | --- |
| [`activation_filter_projection`](activation_filter_projection/) | Projection after activation-based training removal |
| [`activation_filter_refit`](activation_filter_refit/) | Refit the injection direction on the activation-filtered model |
| [`activation_filter_training`](activation_filter_training/) | Activation-ranked training removal |
| [`apps_injection_detection`](apps_injection_detection/) | APPS black-box injection detection |
| [`apps_released_direction`](apps_released_direction/) | Injection-direction projection on the released 4B |
| [`apps_released_monitors`](apps_released_monitors/) | Released monitors on APPS monitor-directed injections |
| [`apps_score_fusion`](apps_score_fusion/) | Combining safety and injection-presence scores |
| [`augmented_judge_evaluation`](augmented_judge_evaluation/) | Augmented adapters on the original JudgeDeceiver holdout |
| [`b200_apps`](b200_apps/) | Current optimized monitor on APPS |
| [`b200_augmented_bf16_apps`](b200_augmented_bf16_apps/) | BF16-trained augmented monitor on APPS |
| [`b200_augmented_bf16_id`](b200_augmented_bf16_id/) | Same augmented checkpoint, BF16 ID control |
| [`b200_augmented_bf16_optimized`](b200_augmented_bf16_optimized/) | Same BF16-trained augmented adapter, optimized ID/APPS serving |
| [`b200_augmented_bf16_training`](b200_augmented_bf16_training/) | Augmented training with BF16 MLPs and FA4 |
| [`b200_augmented_judge`](b200_augmented_judge/) | Same augmented adapter, BF16 and optimized JudgeDeceiver judging |
| [`b200_augmented_sdpa_apps`](b200_augmented_sdpa_apps/) | APPS follow-up for the BF16/SDPA replay |
| [`b200_augmented_sdpa_replay`](b200_augmented_sdpa_replay/) | Historical augmented adapter replay with BF16/SDPA |
| [`b200_augmented_seed_repeat`](b200_augmented_seed_repeat/) | Same-seed BF16/FA4 augmented training repeat |
| [`b200_augmented_training`](b200_augmented_training/) | Augmented-monitor replication with current FP4 training |
| [`b200_historical_augmented_id`](b200_historical_augmented_id/) | Historical augmented adapter on current BF16 ID serving |
| [`b200_historical_augmented_optimized`](b200_historical_augmented_optimized/) | Historical augmented adapter, optimized versus BF16 ID serving |
| [`b200_injection_direction`](b200_injection_direction/) | Small-fit APPS injection-direction replication |
| [`b200_judge_direction_signal`](b200_judge_direction_signal/) | JudgeDeceiver signal versus decision transfer |
| [`b200_judge_numeric_surface`](b200_judge_numeric_surface/) | JudgeDeceiver output-token and candidate-label remapping |
| [`b200_projection_judge`](b200_projection_judge/) | APPS-fitted full projection transferred to JudgeDeceiver |
| [`b200_sdpa_lens`](b200_sdpa_lens/) | Restore Lens on the SDPA-trained monitor |
| [`b200_training_clean_direction`](b200_training_clean_direction/) | Complete original-training direction census |
| [`b200_training_direction`](b200_training_direction/) | Training-example alignment with the APPS direction |
| [`b200_vllm_lens`](b200_vllm_lens/) | vLLM Lens for monitor scoring |
| [`base_judge_steering`](base_judge_steering/) | Augmented-vector steering of base-model JudgeDeceiver judgments |
| [`caft_direction_refit`](caft_direction_refit/) | Model-specific injection projection after regular-data CAFT |
| [`caft_honest_steering`](caft_honest_steering/) | Add the augmented-model vector to CAFT honest examples |
| [`caft_original_projection`](caft_original_projection/) | Original projection at inference on the CAFT checkpoint |
| [`concept_ablation_training`](concept_ablation_training/) | Concept-ablation training on regular monitoring data |
| [`judge_injection_continuation`](judge_injection_continuation/) | Preference continuation through judge-targeted injections |
| [`judge_injection_joint`](judge_injection_joint/) | Joint monitoring and injection-resistant preference training |
| [`monitor_injection_augmentation`](monitor_injection_augmentation/) | Monitor-directed injection augmentation |
| [`monitor_injection_augmentation_9b`](monitor_injection_augmentation_9b/) | Monitor-directed injection augmentation: Qwen3.5-9B replication |
| [`monitoring_injection_disentanglement`](monitoring_injection_disentanglement/) | Injection presence and behavior disentanglement |
| [`monitoring_injection_removal`](monitoring_injection_removal/) | Standard-prompt 4B with aggressive census filtering |
| [`student_injection_awareness`](student_injection_awareness/) | Student prompt-injection awareness |
| [`training_firewall_census`](training_firewall_census/) | Original-training firewall concept census |
| [`training_injection_census`](training_injection_census/) | Training prompt-injection census |
