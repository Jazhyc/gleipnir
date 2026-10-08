# Optimized vLLM serving default and frozen comparisons

The user selects **FP8 full-attention projections on vLLM 0.31** on 2026-10-08
for future inference, serving and evaluation after reviewing the
[projection-precision comparison](../findings/b200_attention_precision.md).
This explicitly supersedes the development triage's BF16 preference. The
current selection is checksum-bound in
`experiments/b200_inference_benchmark/serving_default.json`, resolved by
`gleipnir.serving.reference.selected_serving_default`.

## Current recipe

Use the causal LAST two-logit monitor, native Gigatoken and direct FROST host
bindings, **16 full-attention QKV/output FP8 W8A8 projections**, all 64 FP4 MLP
and 48 FP4 GDN projections, native packed FP4 SwiGLU output and cuDNN MXFP8
full-attention prefill. Keep BF16 projection outputs, cache and recurrence,
FP32 gates/state, prefix caching off, chunked prefill, synchronous stock FCFS,
32,768 context/chunk tokens and runner capacity 128. FP8 uses per-channel
weight scales, dynamic per-token activation scales and native CUTLASS GEMM.
Training precision is a separate selection.

Send complete rendered prompts to `POST /v1/monitor/score`; see the
[endpoint contract](../../experiments/b200_monitor_score/README.md).

Use `experiments.b200_attention_precision.server`, `AttentionPrecisionWorker`
and the source-bound `Vllm031PoolingScheduler` zero-output-token reservation
correction. CUDA GDN/automatic FlashInfer context parallelism remain selected.
Torch 2.13 native compilation checks supersede the old Torch 2.11 mutation
repair; retain that repair in historical 0.24 controls.

The normal startup entrypoint now launches this default:

```bash
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_attention_gdn_serving.startup --name fp8_default01
```

`--prepare-only` validates the selection/runtime/source bindings and emits its
command without allocating GPU memory. `--legacy-fp4` explicitly selects the
archived FP4 startup, including old frontend/host ablation options. Retire the
previous server before an incompatible launch and preserve shared caches.

## Acceptance and envelope

This is `user_accepted_finite` for the evaluated final 272-update FP4/FA4-trained
4B adapter and recipe. Preserve the failed strict master and accepted-0.24
receipts in the [precision finding](../findings/b200_attention_precision.md).
Acceptance does not relabel either failure as parity. Startup must reproduce
the evaluated FP8 canary within MAE 0.005/correlation 0.995, with exact tokens,
finite margins and a nonzero adapter effect; verify actual native dispatch.
This reproduction check does not establish held-out quality parity.

The native envelope is the pinned Qwen3.5/Gleipnir 4B geometry on B200 SM100
through 32K. New adapters require their own FP32-master/merged-serving parity
and nonzero effect; this adapter's accepted drift does not waive their guards.
Other models, hardware or extended contexts require explicit validation.
The earlier 0.24 `LongContextWorker` receipts through 262,144 positions remain
separate historical evidence, not validation of the new FP8 default. Keep
cache-free and length-aware admission alternatives experimental.

## Frozen comparison controls

`experiments/b200_inference_benchmark/baseline.json` still binds the original
0.24 `mutation03` control, every saved repeat and its original failed/accepted
receipts. `selected_score_reference` resolves that immutable comparison;
`selected_serving_default` resolves the current launch recipe. Do not overwrite
old commands, predictions, runtimes, canaries or measurement contracts when
changing the default. Generation comparisons require a named generation control.

The [migration](../findings/b200_vllm031.md),
[same-adapter ID drift](../findings/b200_optimized_id.md),
[precision recovery](../findings/b200_attention_precision.md) and
[long-context](../findings/b200_long_context.md) findings preserve numerical
evidence and qualifications. Standing process, reporting, cache and new-adapter
rules remain in the [inference guide](../agent_guides/inference.md).
