# Serving experiments

Inference and serving: kernels, precision, context, scheduling, throughput. Add new experiments to this table.
Experiments bound by the accepted serving default or the released image stay
directly under `experiments/`; see
[Kept at their recorded paths](../README.md#kept-at-their-recorded-paths).

| Experiment | Question |
| --- | --- |
| [`b200_async_scheduling`](b200_async_scheduling/) | Async scheduling for the pooling scorer |
| [`b200_attention_headroom`](b200_attention_headroom/) | MXFP8 D256 attention headroom on B200 |
| [`b200_context_scaling`](b200_context_scaling/) | Fixed 2K reference scaling |
| [`b200_fp4_inference`](b200_fp4_inference/) | Native FP4 MLP serving on B200 |
| [`b200_gdn_precision`](b200_gdn_precision/) | FP8 GDN projections on the FP8-attention default |
| [`b200_gdn_state`](b200_gdn_state/) | BF16 GDN recurrent-state serving screen |
| [`b200_host_overhead`](b200_host_overhead/) | Host overhead of short single requests |
| [`b200_inference_kernels`](b200_inference_kernels/) | B200 inference kernel screen |
| [`b200_kernel_profile`](b200_kernel_profile/) | Kernel profile of the adopted last-query default |
| [`b200_lean_dispatch`](b200_lean_dispatch/) | Lean dispatch for gleipnir custom ops (not adopted) |
| [`b200_length_admission`](b200_length_admission/) | Length-aware admission for monitoring scores |
| [`b200_long_context`](b200_long_context/) | Single-request long-context serving |
| [`b200_optimized_id`](b200_optimized_id/) | Optimized-stack ID score drift |
| [`b200_score_graphs`](b200_score_graphs/) | Large prefill graphs on the repaired monitoring endpoint |
| [`b200_score_scaling`](b200_score_scaling/) | Selected reference concurrency scaling |
| [`b200_swiglu_batching`](b200_swiglu_batching/) | Batch-composition dependence of monitor scores |
| [`b200_swiglu_output`](b200_swiglu_output/) | Disable direct FP4 SwiGLU output |
| [`classifier_throughput`](classifier_throughput/) | Direct-classifier inference throughput |
| [`fp4_inference`](fp4_inference/) | Blackwell vLLM inference search |
| [`gleipnir4b_serving_ladder`](gleipnir4b_serving_ladder/) | Serving ladder for Gleipnir 4B |
| [`local_inference`](local_inference/) | Local merged Gleipnir 4B baseline |
| [`optimized_gap`](optimized_gap/) | Optimized-serving gap for the Gleipnir release candidates |
| [`prefix_caching_replay`](prefix_caching_replay/) | Per-action monitoring with prefix caching on the released stack |
| [`prefix_caching_tokens`](prefix_caching_tokens/) | Prompt work of monitoring every action |
| [`qwen38_27b_serving`](qwen38_27b_serving/) | Arithmetic-preserving serving optimizations for Qwen3.8-27B |
| [`runpod_gleipnir4b_id`](runpod_gleipnir4b_id/) | Runpod B200 setup and Gleipnir 4B ID inference |
| [`serving_concurrency`](serving_concurrency/) | Serving throughput and latency by step and concurrency |
