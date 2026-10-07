# Optimized vLLM serving default and score reference

User-selected as the serving reference on 2026-10-07 and as the default for
inference, serving and evaluation in **all future experiments** on 2026-10-08.
Use the cached **two-logit monitoring endpoint with validated Triton mutation
analysis**, native Gigatoken and direct FROST host bindings for monitor scoring.
Preserve the selected Direct FP4 MLP/projection recipe, symbolic-row SwiGLU and
direct packed FP4 activation output, cuDNN MXFP8 full-attention prefill, BF16
recurrence/cache and FP32 gates/state. Prefix caching remains off; chunked
prefill and causal LAST pooling remain on. This selects the inference backend;
the training recipe remains governed separately.

The audited native kernels currently target the recorded Qwen3.5/Gleipnir 4B
projection shapes and B200 SM100 runtime. For another model, GPU or task,
establish compatibility and validate the needed extension before a large run;
record any backend exception explicitly. New adapters use their own merged
serving artifact and master/serving score canary with a nonzero adapter effect.
Reuse unchanged native receipts and persistent compiler/kernel caches. The
current adapter's accepted numerical drift does not establish other adapters'
quality or justify modifying frozen past evaluation contracts. The full
[same-adapter ID comparison](../findings/b200_optimized_id.md) shows material
held-out drift despite reproducing the accepted small canary; small-canary
agreement alone cannot support an ID quality-parity claim.

Use the source-bound `PoolingContextScheduler` correction from
`experiments/b200_long_context/scheduler.py` for chunked pooling: reserve zero
generated tokens so exact-cap prompts can finish. It preserves synchronous
FCFS order. For contexts through 32K, use the audited `MonitorScoreAuditWorker`;
for longer inputs, use the validated `LongContextWorker` envelope up to the
checkpoint's 262,144 positions. Keep prefill chunks at 32,768 tokens, within the
FP4 kernels' validated row envelope. Bind the scheduler and long-context native
receipts as recorded in
[the long-context experiment](../../experiments/b200_long_context/README.md).
The 256K result establishes C1 systems capacity; long-context model quality and
multi-request 256K capacity are unmeasured. Keep the cache-free path and rejected
length-aware admission experimental; neither replaces this cached FCFS default.

`experiments/b200_inference_benchmark/baseline.json` binds `mutation03`, every
saved prediction repeat, classifier/precision/compiler/native receipts and the
user's finite-quality acceptance. It remains the immutable 32K optimization
comparison control; the broader default and later context/correctness receipts
do not overwrite its original command or predictions. The previous generation
selection is archived in
`baselines/frost_fp4_gigatoken_direct_host_generation.json`. Use
`gleipnir.serving.reference.selected_score_reference` to validate the selection.
Generation experiments must name their archived control explicitly; their
request protocol cannot be used with this pooling worker.

The endpoint is `POST /v1/monitor/score` with the complete rendered text prompt;
see [its contract](../../experiments/b200_monitor_score/README.md). Startup uses
`experiments.b200_mutation_analysis.server`, `MonitorScoreAuditWorker` and the
pinned mutation validation receipt. Reconstruct the source/runtime/environment
from the selected server metadata and preserved caches, then apply the chosen
validated context worker and pooling correction with explicit source bindings.
Never remove the repair or replace pooling with generation silently. New
compatible score comparisons
use the shared score runner and `baseline: selected`; native repair gates may be
reused while their source/runtime identities remain unchanged.

Reuse three c1/quick64 and six c128/full320 warm repeats on EU-RO-1:
**31.23 ms median / 128.54 ms p95**, **215707 input tokens/s**. Source-macro/pooled
AUROC is **0.873931/0.887170**. This is a frozen training-seen systems-development
cohort, not held-out quality or production traffic. Preserve strict master and
native precision failures separately from the accepted finite tradeoff. The
[concurrency](../findings/b200_score_scaling.md),
[fixed 2K](../findings/b200_context_scaling.md) and
[long-context](../findings/b200_long_context.md) findings retain later timings
and their distinct workload/quality qualifications.

The mutation repair changes compiler analysis, not the Triton runtime kernel.
Isolated profiles do not establish a kernel speedup; sequential end-to-end
measurements include batch schedule and host variation. See the
[mutation analysis finding](../findings/b200_triton_mutation_analysis.md) for
validation, score/AUROC differences and failed attempts. Retain a single warm
server and persistent caches; stop it before incompatible graph/kernel changes.
