# Systems experiments

Training throughput, kernels, precision and compilation on B200. Add new experiments to this table.

| Experiment | Question |
| --- | --- |
| [`b200_adaptive_microbatching`](b200_adaptive_microbatching/) | Adaptive B200 physical microbatches |
| [`b200_bf16_fa4`](b200_bf16_fa4/) | FlashAttention 4 on the packed BF16 B200 recipe |
| [`b200_fouroversix`](b200_fouroversix/) | MLP-only Four Over Six training on B200 |
| [`b200_fp4_full_training`](b200_fp4_full_training/) | Full FP4 training replication and ID evaluation |
| [`b200_fp4_mlp_lora`](b200_fp4_mlp_lora/) | Native FP4 MLPs with BF16 attention and FP32 LoRA on B200 |
| [`b200_frost_training`](b200_frost_training/) | Direct FROST bindings for FP4 LoRA training |
| [`b200_meta_stack`](b200_meta_stack/) | Meta optimization campaign for causal D256 GQA |
| [`b200_mlp_gemm`](b200_mlp_gemm/) | B200 MLP/GEMM optimization |
| [`b200_mxfp8_fused`](b200_mxfp8_fused/) | Fused MXFP8 operand preparation |
| [`b200_nvidia_mxfp8`](b200_nvidia_mxfp8/) | NVIDIA MXFP8 full-attention training screen |
| [`b200_nvidia_mxfp8_varlen`](b200_nvidia_mxfp8_varlen/) | Direct packed NVIDIA MXFP8 attention |
| [`b200_training_throughput`](b200_training_throughput/) | B200 training throughput: ten-update iteration |
| [`b200_training_trim`](b200_training_trim/) | Final-layer trim in training |
| [`monitoring_adamw_throughput`](monitoring_adamw_throughput/) | Monitoring AdamW throughput screen |
| [`monitoring_branching`](monitoring_branching/) | Gradient-preserving prompted branches |
| [`monitoring_gradient_checkpointing`](monitoring_gradient_checkpointing/) | Monitoring gradient-checkpointing screen |
| [`monitoring_inner_mixer_compile`](monitoring_inner_mixer_compile/) | Monitoring inner-mixer compilation |
| [`monitoring_linear_shell_compile`](monitoring_linear_shell_compile/) | Monitoring linear-attention shell compilation |
| [`monitoring_partial_checkpointing`](monitoring_partial_checkpointing/) | Monitoring partial checkpointing |
| [`monitoring_selective_checkpointing`](monitoring_selective_checkpointing/) | Monitoring selective-checkpointing screen |
| [`monitoring_selective_compile`](monitoring_selective_compile/) | Monitoring selective-compilation screen |
| [`monitoring_sequence_packing`](monitoring_sequence_packing/) | Independent monitoring sequences in packed Qwen3.5 training |
| [`monitoring_training_throughput`](monitoring_training_throughput/) | Monitoring-training throughput screen |
| [`qwen38_27b_training_speed`](qwen38_27b_training_speed/) | Qwen3.8-27B training speed on the ten-update benchmark |
