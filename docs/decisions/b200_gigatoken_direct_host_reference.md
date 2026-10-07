# Native Gigatoken and direct FROST serving reference

User-selected on 2026-10-07. The B200 inference optimization reference is the
existing Direct FP4 GPU recipe with **native Gigatoken 0.10.0 encoding and direct
FROST host binding reuse**. Keep original CPU affinity, FP4 projection/MLP scopes,
MXFP8 causal prefill, BF16 recurrence/cache/decode and FP32 gates/state, together
with inherited strict numerical failures and finite acceptance. No training
recipe changes.

`experiments/b200_inference_benchmark/baseline.json` binds the combined reference
and both CPU-component receipts by checksum. Archive the preceding selection
under `baselines/frost_fp4_mlp_gdn_attention_mxfp8_native_swiglu_fp4_output.json`.
The source GPU configuration and validation remain the recorded parent;
CPU components do not invalidate GPU compiler caches.

Reuse the completed direct-mode measurements in `host_wrapper_compare01`.
The derived `results/b200_attention_gdn_serving/gigatoken_direct_host_reference01`
contains three c1/quick64 repeats, six c128/full320 repeats, original predictions,
startup parity and reconstruction provenance. No new timing or numerical run is
claimed during selection. Controls are **149.69 ms median, 184.19 ms p95 and
197530 input tokens/s at c128**. Source-macro/pooled development AUROC is
0.878144/0.885763. These are training-seen systems-development workloads, not
production traffic or final ID.

These are historical NC2 timings. On EU-RO-1, the same selected cached recipe
measures **30.36 ms c1 median** and **211041 input tokens/s** in the matched
`results/b200_attention_gdn_serving/cache_policy_euro02/fresh_cached` phase.
Use its three c1/six c128 repeats for performance attribution on that host;
retain the selected archive for its frozen recipe and numerical provenance.
Do not attribute cross-host latency differences to a kernel intervention.
The user retains this standard cached path after the comparison: the cache-free
prototype's roughly 5-ms median saving does not justify its observed throughput
tradeoff for this workload. Keep prefix caching disabled in matched timing
comparisons; KV/state storage for chunked prefill is a separate feature.
See the [same-host finding](../findings/b200_cache_policy_same_host.md).

Future selected comparisons use c1 and c128 and all available baseline repeats.
Other combined-stack concurrency controls are unmeasured; do not substitute
historical HF timings or mix quick64 with full320. The wrapper's paired gain was
0.60% with five of six positive pairs, but its interval includes zero.

Standard startup selects both CPU components automatically:

```bash
PYTHONPATH=src:. python -m experiments.b200_attention_gdn_serving.startup --name unique_start
```

Retire the old API/engine before replacement; compatible trials reuse the warm
worker. `--legacy-host` explicitly requests HF encoding and original bindings
for a named diagnostic. The selected optimization reference retains the control
instrumentation of the measured process, bound in server metadata. Its original
selection reused the resident worker without a restart or cold cache. Inspect
`results/b200_attention_gdn_serving/server.json` for current process identity;
historical PIDs remain in the finding records.

Focused default-selection, receipt-drift and resident-command tests pass.
See the [encoder finding](../findings/b200_gigatoken_frontend.md) and
[wrapper finding](../findings/b200_frost_host_wrappers.md) for source/version,
exact token/native parity, quality diagnostics and artifact details.
