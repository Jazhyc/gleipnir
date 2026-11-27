# Decisions

Every decision, newest first. Each lives in its research-area folder; the
[current state](../current.md) lists the ones in force. Entries carried over
from the former single-file decision log say so in their `Status:` line.

| Date | Area | Decision |
| --- | --- | --- |
| 2026-11-23 | evaluation | [Inkling 2 Mini as the next large-monitor base](evaluation/inkling2_mini_next_base.md) |
| 2026-10-20 | training | [Train Gleipnir 4B on tool-trajectory data only](training/train_gleipnir_4b_on_tool_trajectory_data_only.md) |
| 2026-10-20 | serving | [Price Gleipnir release models at self-hosted B200 break-even](serving/price_gleipnir_release_models_at_self_hosted_b200_break_even.md) |
| 2026-10-20 | evaluation | [Plot Gleipnir at optimized-serving performance; no QAD for the 27B](evaluation/plot_gleipnir_at_optimized_serving_performance_no_qad_for.md) |
| 2026-10-20 | evaluation | [Evaluate Gleipnir 27B under the full teacher prompt](evaluation/evaluate_gleipnir_27b_under_the_full_teacher_prompt.md) |
| 2026-10-17 | systems | [Qwen3.8-27B B200 training default](systems/qwen38_27b_b200_training_default.md) |
| 2026-10-16 | serving | [Opt-in prefix-cached serving](serving/prefix_cached_serving.md) |
| 2026-10-16 | systems | [Native FP4 MLP B200 training default](systems/b200_native_fp4_training_recipe.md) *(superseded)* |
| 2026-10-16 | systems | [BF16/SDPA B200 training default](systems/b200_bf16_sdpa_training_default.md) |
| 2026-10-15 | evaluation | [Precision and kernel choices may be selected on the OOD suite](evaluation/precision_selection_on_ood.md) |
| 2026-10-15 | serving | [Gate the GDN context-parallel prefill route by length](serving/gate_the_gdn_context_parallel_prefill_route_by_length.md) |
| 2026-10-15 | serving | [Adopt the lean GDN host path with a strided gated norm](serving/adopt_the_lean_gdn_host_path_with_a_strided_gated_norm.md) |
| 2026-10-15 | serving | [Adopt last-query final-layer attention in the FP8 default](serving/adopt_last_query_final_layer_attention_in_the_fp8_default.md) |
| 2026-10-13 | quantization | [Selected merged-weight QAT training recipe](quantization/qat_reference_training.md) |
| 2026-10-12 | evaluation | [Retain the frozen OOD inputs for paper comparability](evaluation/frozen_ood_comparability.md) |
| 2026-10-11 | teachers | [Strict teacher agreement is the default training-data filter](teachers/strict_teacher_filtering.md) |
| 2026-10-08 | serving | [Monitor serving precision and frozen comparisons](serving/b200_monitor_score_reference.md) |
| 2026-10-08 | serving | [Default future experiments to optimized vLLM](serving/default_future_experiments_to_optimized_vllm.md) |
| 2026-10-07 | serving | [Native Gigatoken and direct FROST serving reference](serving/b200_gigatoken_direct_host_reference.md) *(superseded)* |
| 2026-10-07 | serving | [Direct FP4-output B200 inference reference](serving/b200_native_fp4_output_inference_baseline.md) |
| 2026-10-06 | serving | [Reuse training FROST forward arithmetic in vLLM](serving/b200_frost_inference.md) |
| 2026-10-06 | serving | [Production inference development benchmark](serving/b200_inference_benchmark.md) |
| 2026-10-06 | serving | [Native inference kernel screening on B200](serving/b200_inference_kernels.md) |
| 2026-10-06 | serving | [Native cuDNN FP4 MLP inference screen](serving/b200_fp4_inference.md) |
| 2026-10-06 | serving | [MXFP8 prefill as the B200 inference optimization reference](serving/b200_mxfp8_inference_baseline.md) *(superseded)* |
| 2026-10-06 | systems | [Latest-only weights for short systems training checks](systems/systems_training_adapter_retention.md) |
| 2026-10-06 | serving | [FP4 attention projections as the B200 inference reference](serving/b200_attention_fp4_inference_baseline.md) |
| 2026-10-06 | serving | [FP4 GDN projections as the B200 serving baseline](serving/b200_fp4_gdn_inference_baseline.md) *(superseded)* |
| 2026-10-06 | serving | [Combined preparation and tuned FP4 GEMMs as the B200 inference reference](serving/b200_tuned_fp4_inference_baseline.md) *(superseded)* |
| 2026-10-06 | serving | [B200 serving precision after FROST baseline selection](serving/b200_low_precision_attention_serving.md) |
| 2026-10-05 | systems | [Select native FP4 MLPs as the B200 training default](systems/select_native_fp4_mlps_as_the_b200_training_default.md) |
| 2026-10-05 | systems | [Resident B200 FP4 optimization baseline](systems/b200_fp4_optimization_worker.md) |
| 2026-10-05 | systems | [Full regular 4B replication with native FP4 MLPs](systems/b200_fp4_full_training_replication.md) |
| 2026-10-04 | injection | [Replicate monitor-directed injection augmentation on 9B](injection/monitor_injection_augmentation_9b.md) |
| 2026-10-04 | systems | [Packed BF16 LoRA as the B200 training default](systems/b200_packed_bf16_training_recipe.md) |
| 2026-10-04 | injection | [Monitor-directed injection replacement](injection/monitor_injection_augmentation.md) |
| 2026-10-03 | systems | [Persistent compiler caches across training campaigns](systems/persistent_training_compiler_cache.md) |
| 2026-10-02 | systems | [Select uniform FlashQLA for B200 NF4/BF16 LoRA](systems/select_uniform_flashqla_for_b200_nf4_bf16_lora.md) |
| 2026-10-02 | systems | [FlashQLA for the default B200 LoRA recipe](systems/b200_flashqla_training_recipe.md) |
| 2026-10-01 | systems | [Adaptive batching for future B200 training](systems/b200_adaptive_training_recipe.md) |
| 2026-09-08 | evaluation | [Use CoT-removed ID inputs for future validation](evaluation/cot_removed_id_validation.md) |
| 2026-09-04 | systems | [Config-driven monitoring systems screens](systems/config_driven_systems_screens.md) |
| 2026-09-04 | training | [Advance `2e-5` for monitoring-only LR confirmation](training/advance_2e_5_for_monitoring_only_lr_confirmation.md) |
| 2026-09-03 | systems | [Enable pinned causal-conv1d for long Qwen3.5-4B training](systems/enable_pinned_causal_conv1d_for_long_qwen3_5_4b_training.md) |
| 2026-08-30 | training | [Run tool-trajectory scaling as Kimi-soft-only distillation](training/run_tool_trajectory_scaling_as_kimi_soft_only_distillation.md) |
| 2026-08-30 | teachers | [Make paid teacher caching provider-sticky and auditable](teachers/make_paid_teacher_caching_provider_sticky_and_auditable.md) |
| 2026-08-17 | training | [Do not promote hard-only from the internal split](training/do_not_promote_hard_only_from_the_internal_split.md) |
| 2026-08-17 | evaluation | [Default frozen text evaluation to vLLM](evaluation/default_frozen_text_evaluation_to_vllm.md) |
| 2026-08-14 | training | [Screen trajectory, regularization, and data weighting](training/screen_trajectory_regularization_and_data_weighting.md) |
| 2026-08-14 | training | [Screen structural training interventions](training/screen_structural_training_interventions.md) |
| 2026-08-14 | training | [Retain the baseline after the regularization screen](training/retain_the_baseline_after_the_regularization_screen.md) |
| 2026-08-13 | training | [Retain AdamW after full-data Muon confirmation](training/retain_adamw_after_full_data_muon_confirmation.md) |
| 2026-08-13 | training | [Confirm the two Muon candidates on full data](training/confirm_the_two_muon_candidates_on_full_data.md) |
| 2026-08-13 | training | [Compare Muon through scheduled update-RMS matching](training/compare_muon_through_scheduled_update_rms_matching.md) |
| 2026-08-13 | training | [Advance two Muon rates to replicated confirmation](training/advance_two_muon_rates_to_replicated_confirmation.md) |
| 2026-08-12 | training | [Measure data and adapter capacity jointly](training/measure_data_and_adapter_capacity_jointly.md) |
| 2026-08-11 | training | [Train text-only LoRAs causally and export serving copies](training/train_text_only_loras_causally_and_export_serving_copies.md) |
| 2026-08-11 | training | [Require FLA and standard QLoRA for the rank curve](training/require_fla_and_standard_qlora_for_the_rank_curve.md) |
| 2026-08-10 | training | [Match multimodal training and serving architectures](training/match_multimodal_training_and_serving_architectures.md) |
| 2026-08-09 | operations | [Bootstrap](operations/bootstrap.md) |
