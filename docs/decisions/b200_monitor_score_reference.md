# Repaired monitoring score reference

User-selected on 2026-10-07. Use the cached **two-logit monitoring endpoint with
validated Triton mutation analysis**, native Gigatoken and direct FROST host
bindings as the B200 inference optimization reference. Preserve the selected
Direct FP4 MLP/projection recipe, cuDNN MXFP8 full-attention prefill, BF16
recurrence/cache and FP32 gates/state. Prefix caching remains off; chunked
prefill and causal LAST pooling remain on. Training is unchanged.

`experiments/b200_inference_benchmark/baseline.json` binds `mutation03`, every
saved prediction repeat, classifier/precision/compiler/native receipts and the
user's finite-quality acceptance. The previous generation selection is archived
in `baselines/frost_fp4_gigatoken_direct_host_generation.json`. Use
`gleipnir.serving.reference.selected_score_reference` to validate the selection.
Generation experiments must name their archived control explicitly; their
request protocol cannot be used with this pooling worker.

The endpoint is `POST /v1/monitor/score` with the complete rendered text prompt;
see [its contract](../../experiments/b200_monitor_score/README.md). Startup uses
`experiments.b200_mutation_analysis.server`, `MonitorScoreAuditWorker` and the
pinned mutation validation receipt. Reconstruct the source/runtime/environment
from the selected server metadata and preserved caches; never remove the repair
or replace pooling with generation silently. New compatible score comparisons
use the shared score runner and `baseline: selected`; native repair gates may be
reused while their source/runtime identities remain unchanged.

Reuse three c1/quick64 and six c128/full320 warm repeats on EU-RO-1:
**31.23 ms median / 128.54 ms p95**, **215707 input tokens/s**. Source-macro/pooled
AUROC is **0.873931/0.887170**. This is a frozen training-seen systems-development
cohort, not held-out quality or production traffic. Preserve strict master and
native precision failures separately from the accepted finite tradeoff; future
adapters still require their own parity checks.

The mutation repair changes compiler analysis, not the Triton runtime kernel.
Isolated profiles do not establish a kernel speedup; sequential end-to-end
measurements include batch schedule and host variation. See the
[mutation analysis finding](../findings/b200_triton_mutation_analysis.md) for
validation, score/AUROC differences and failed attempts. Retain a single warm
server and persistent caches; stop it before incompatible graph/kernel changes.
