# Inference and evaluation instructions

These instructions are required for the tasks routed here by the root
[AGENTS.md](../../AGENTS.md). Standing preferences were moved from that
file on 2026-10-07; their scope and historical acceptance remain intact.

Also read [compute.md](compute.md) before launching or operating a run.

## Serving and adapter parity

Standing user preference, 2026-10-08: use the optimized vLLM implementation for
inference, serving and evaluation in all future experiments. Monitor scoring
uses the cached causal two-logit path in the
[serving decision](../decisions/b200_monitor_score_reference.md), including its
validated context handling and pooling boundary correction. Select this path
explicitly in new experiment launch configurations; the decision defines the
recipe and supported hardware/model envelope. Reuse a persistent engine and
continuous batching rather than an eager training microbatch.

New adapters and backend/layout combinations require master-to-serving score
agreement and a nonzero adapter effect before scaling evaluation. Reuse unchanged
native/kernel receipts and persistent caches. Record unsupported model/task/GPU
combinations and any required backend exception explicitly; use Transformers
eager for bounded parity diagnostics or unsupported adapters. Preserve frozen
past evaluation contracts and their named controls. Training precision and
optimization are governed by the separate training guide.

Standing user preference, 2026-10-08: create an adapter-specific merged BF16
checkpoint for every trained adapter used in inference, serving or evaluation,
serving without dynamic adapter projections.
Keep FP32 master adapters and pinned base weights persistent. On Runpod, store
reconstructable merged checkpoints on ephemeral container storage, with merge
source/file checksums and evaluation receipts on the network volume. Accumulate
the merge in FP32 before exporting BF16 and run adapter-specific serving parity
before scaling evaluation. Record unsupported layouts or failed parity explicitly;
do not silently fall back or change frozen historical evaluation contracts.
Changing adapters requires loading the corresponding merged model; preserve disk
compiler caches across those restarts. See the merged serving protocol in
`experiments/b200_inference_benchmark/README.md`.

If a training repeat reproduces the exact master/export weight hashes and
equivalent adapter/base identities, reuse the existing merged checkpoint,
parity and evaluation receipts. Record identity and score reuse explicitly;
repeat scoring only when measuring serving variability or changing its recipe.

## Persistent serving process

Standing user preference: when an inference change needs a different
serving process, stop the old server rather than keeping both resident. Preserve
its measured baseline results, merged checkpoint, logs and persistent caches.
Reuse completed matched baseline results; do not retain or rerun a control merely
because a new kernel is being tested. Keep the active candidate warm for compatible
trials. After a rejected or failed kernel trial, retire its worker and leave the
GPU available for the next change; do not automatically restart the reference
server. Restore a reference only when the user requests it or the next task needs
that process. This is process replacement, not authorization to terminate capacity.

## Optimization reporting

Standing user preference, 2026-10-06: inference optimization reports must include
prompt tokens per second alongside requests per second and request latency.
Use prompt-token throughput as the main compute-throughput comparison for the
one-token monitor, keeping the prompt-length distribution and cache policy fixed.
Label input/prompt throughput explicitly rather than confusing it with generated
output-token throughput.

Standing user preference, 2026-10-06: every inference kernel update must report
AUROC deviation against the same frozen baseline, alongside speed and score
parity. Include pooled/per-source AUROC and macro across sources with both
labels, repeat variation, explicit undefined single-label sources, and bind
labels to exact prompt identities. Reuse archived baseline predictions rather
than rerunning the control. Small training-seen optimization sets are diagnostic
and do not establish held-out quality parity or select production precision.

## B200 comparison controls

Read the [serving decision](../decisions/b200_monitor_score_reference.md) and
`experiments/b200_inference_benchmark/baseline.json` before preparing comparisons.
Its checksum-bound controls supersede the October 6 GDN-only reference below;
the future-experiment default does not rewrite these frozen measurements.
Compare timing and quality on matched concurrency/cohorts using all available
repeats. The [concurrency](../findings/b200_score_scaling.md),
[fixed 2K](../findings/b200_context_scaling.md) and
[long-context](../findings/b200_long_context.md) records define their additional
measurement contracts. Preserve failed strict checks separately from accepted
finite quality. Acceptance remains adapter/recipe specific and does not waive
finite-output checks, new-adapter parity or held-out evaluation requirements.

Keep the current launch selection and frozen comparison bindings separate, as
defined in the linked serving decision;
avoid duplicating changing result paths, measurements or worker PIDs here.

## Historical GDN baseline acceptance, 2026-10-06

The following preserves the earlier standing preference and its numerical
acceptance. Its selected control and BF16 full-attention recipe are historical;
new comparisons use the current reference above.

Standing user preference, 2026-10-06 (updated): use FROST native FP4 MLPs
and all 48 large GDN QKV/Z and output projections as the B200 inference
optimization baseline. Keep small gate projections, convolution and recurrence
operands BF16, gates/state FP32, and full attention BF16 FlashInfer. The user
explicitly accepts the c128 source-macro AUROC drop of 0.47 percentage points;
retain the strict failed score-canary receipt separately as `user_accepted_finite`.
The selected control is `results/b200_attention_gdn_serving/fp4_gdn_projection02`,
with five-pass warmed confirmation at `fp4_gdn_confirmation01`, bound by
`experiments/b200_inference_benchmark/baseline.json`. Prioritize c128 throughput,
retaining peak throughput, lower-concurrency latency and AUROC diagnostics.
New conditions compare against this selected baseline; preserve explicit
historical references. Reuse archived controls and persistent caches. Stop the
old server before changing active kernels so the candidate gets the whole GPU.
Keep BF16 GDN/MLP and FP8 comparisons as historical context. This acceptance
applies to the current adapter and pinned serving recipe; it changes neither
training nor frozen past evaluations and does not waive missing/nonfinite
outputs or adapter-specific checks for future layouts/adapters.
