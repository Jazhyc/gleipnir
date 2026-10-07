# BF16 GDN recurrent-state serving screen

Hypothesis: BF16 persistent recurrent state and native BF16 FlashInfer state I/O
reduce state traffic/conversion overhead while retaining usable monitor scores.
Use the existing B200, frozen merged adapter and selected repaired score endpoint.
Keep all FP4 projections/MLPs, MXFP8 full attention, gates, FP32 MMA accumulation,
convolution, prompt identities, causal LAST pooling and cache policy fixed.
No backend upgrade or MLP optimization is part of this screen.

The pinned FlashInfer 0.6.12 SM100 adapter accepts BF16 state beneath its public
FP32 documentation. vLLM's ordinary wrapper upcasts state to FP32; the candidate
passes BF16 initial/final buffers explicitly and sets `--mamba-ssm-cache-dtype
bfloat16`. Internal accumulation remains FP32. The Qwen model advertises FP32
state, so its expected explicit-override warning must be recorded. Audit actual
cache tensors on all 24 GDN layers; do not infer storage from model metadata.

Freeze native output/state relative-L2 limits at 3%. Check lengths 1,17,129,
ragged [1,127,513],4096,[8192,8192],32768; nonzero state, strided inputs, ragged
isolation, a sequential FP32 oracle, continued state and changed-input CUDA-graph
replay. Separately round forget/update gates to BF16 and convert them back to
the FP32 native boundary as a numerical diagnostic, not a speed candidate.
True BF16 MMA accumulation is unsupported by this pinned kernel and is not
represented by that diagnostic. Preserve all failures and source/runtime hashes.

Retire the sole identity-verified serving process before native checks; preserve
its source/environment identity, logs, checkpoint and shared compiler caches.
On failure restore the selected reference without replaying its timing controls.
After native admission, require the existing 20-row score canary (mean absolute
error <=0.005, correlation >=0.995, nonzero adapter effect), finite outputs and
live BF16 state dispatch/cache coverage. Compare three c1/quick64 and six
c128/full320 warm repeats against every archived selected repeat, after one
excluded warmup at each concurrency. Report input tokens/s, request latency,
pooled/per-source/source-macro AUROC in percentage points, calibration, ties
and threshold flips. This is training-seen systems development, not held-out
quality evidence. No final-ID evaluation or automatic promotion.

A useful candidate needs >1% median c128 throughput gain, <=2% c1 median/p95
regression and absolute macro/pooled AUROC shifts <=0.1 percentage points.
This screening rule is diagnostic; retain strict/inherited precision failures.
Restore the reference after a rejected candidate. Stop on provenance/input drift,
native/canary/audit failure, nonfinite output, server failure or bounded completion.
No capacity creation or termination. Startup is monitored in the active turn;
no after-turn heartbeat scheduler is available in this session.

```bash
PYTHONPATH=src:. python -m experiments.b200_gdn_state.run --name bf16_state01
```

An exited-startup retry may use `--retired-parent <parent_server.json>` and
`--reuse-native <native.json>`. Reconstruct the staged environment from public
runtime/frontend/cache receipts, verify their identities, and reuse native
admission only while every bound source hash still matches. An already-exited
candidate is archived only after confirming that no GPU worker remains.

Artifacts: `results/b200_gdn_state/<name>/`; runtime logs remain under
`logs/runpod/b200_attention_gdn_serving/`. The selected reference remains
checksum-bound by `experiments/b200_inference_benchmark/baseline.json`.
