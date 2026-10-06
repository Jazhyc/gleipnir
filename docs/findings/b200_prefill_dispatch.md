# Prefill launch overhead and CUDA-graph coverage

Investigated 2026-10-07 on the saved Direct FP4 reference trace and archived
installed vLLM 0.24.0 source. No new GPU run, server restart or recipe promotion.
The direct-GDN output path remains deferred; use the selected reference for
future launch-optimization comparisons.

## Concrete configuration mismatch

The reference enables `FULL_AND_PIECEWISE` graphs, but its capture sizes stop
at 256 tokens. The throughput profile contains 39 batches of 32,768 tokens,
one of 29,116, and two of 1,428/2,085. All exceed the capture envelope.

Installed `CudagraphDispatcher.dispatch` returns `NONE` when
`num_tokens > max_cudagraph_capture_size`. Independently, the saved trace
contains no CUDA graph runtime/driver calls. It contains exactly 44,784
individual launches matching 44,784 kernels: 13,116 `cuLaunchKernelEx`,
8,820 `cuLaunchKernel`, 15,456 `cudaLaunchKernel` and 7,392
`cudaLaunchKernelExC`. Thus configured graph mode does not provide graph
replay for this measured large-prefill workload. Compiled graphs and cached
native kernels do not by themselves eliminate per-kernel host submission.

Source archive: `results/b200_attention_gdn_serving/prefill_dispatch_sources01`,
with compilation config, dispatcher, GPU model runner, CUDA-graph wrapper and
piecewise backend, each hashed. The trace is the already-bound
`native_fp4_output_profile01`; original config is in the collected server log.

## Candidate priorities

1. Extend piecewise graph capture to representative prefill token buckets,
   including 32,768. Replay eligible GEMM/packing/elementwise segments while
   keeping GDN and full attention eager under their existing splitting ops.
   vLLM's default splitting list already includes `qwen_gdn_attention_core`
   and `unified_attention_with_output`. Installed Mamba validation explicitly
   avoids capping piecewise-prefill graph sizes merely because decode requests
   need state-cache blocks. Full graph capture is a separate compatibility task.
2. Reuse activation workspaces and descriptor preparation where feasible.
   Frozen weights, compiled native plans and shared disk caches are already
   reused; target remaining per-call plumbing, not those established caches.
3. Consider a score-only serving path returning the two decision logits.
   One-token scoring already avoids a sustained decode loop, and sampling
   GPU work is only 0.11% in this trace. This is a broader architecture change
   with unmeasured CPU benefit, lower priority than missing graph coverage.

Piecewise capture requires persistent buffers, graph-safe custom operators
and bounded extra memory. Bucket padding can change selected FP4 producer
branches around 1,536/4,096 rows and GEMM tiles, so check finite/padded-row
isolation, pointer/replay correctness and matched score/AUROC before accepting
a candidate. Capture memory and padding cost can offset dispatch savings.
No compatibility or speedup is established by this source inspection.

Removing all observed 7--8% launch-related gaps would imply approximately
8% kernel-window throughput headroom, assuming GPU work unchanged. That is
an optimistic timing bound, not an expected gain: piecewise capture cannot
eliminate eager attention/GDN, all synchronization or required preparation,
and the gaps were measured under profiling. Interactive-latency headroom
requires its own measurements.

A prefill scorer still needs batching and state/KV handling for prompts split
across chunks. Disabling chunked prefill or deleting decode machinery alone
does not address launches inside the model's execution annotations.

Official references: [CUDA-graph design](https://docs.vllm.ai/en/latest/design/cuda_graphs/)
and [v0.24 optimization guide](https://docs.vllm.ai/en/v0.24.0/configuration/optimization/).
Their general guidance is checked against the pinned installed dispatcher;
latest documentation does not imply an upgrade or changed supported recipe.
