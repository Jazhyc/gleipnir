# Inference and evaluation instructions

These instructions are required for the tasks routed here by the root
[AGENTS.md](../../AGENTS.md). Standing preferences were moved from that
file on 2026-10-07; their scope and historical acceptance remain intact.

Also read [compute.md](compute.md) before launching or operating a run.

## Serving and adapter parity

For frozen text-only evaluation of standard base or PEFT LoRA models, default to
one persistent vLLM engine with continuous batching, a constrained one-token
response, and explicitly requested decision-token logprobs. Do not carry an
eager training microbatch into large evaluation runs. Use Transformers eager
evaluation only for a bounded backend-parity canary or when vLLM cannot represent
the model/adapter, and document that exception. For every new adapter layout or
backend combination, compare the master checkpoint with its serving artifact and
record score agreement plus a nonzero adapter effect before scaling evaluation.

Standing user preference, 2026-10-06: merge standard LoRA updates into BF16 base
weights for future evaluations, serving without dynamic adapter projections.
Keep FP32 master adapters and pinned base weights persistent. On Runpod, store
reconstructable merged checkpoints on ephemeral container storage, with merge
source/file checksums and evaluation receipts on the network volume. Accumulate
the merge in FP32 before exporting BF16 and run adapter-specific serving parity
before scaling evaluation. Record unsupported layouts or failed parity explicitly;
do not silently fall back or change frozen historical evaluation contracts.
Changing adapters requires loading the corresponding merged model; preserve disk
compiler caches across those restarts. See the merged serving protocol in
`experiments/b200_inference_benchmark/README.md`.

## Persistent serving process

Standing user preference, 2026-10-06: when an inference change needs a different
serving process, stop the old server rather than keeping both resident. Preserve
its measured baseline results, merged checkpoint, logs and persistent caches.
Reuse completed matched baseline results; do not retain or rerun a control merely
because a new kernel is being tested. Keep the active candidate warm for compatible
trials. This is process replacement, not authorization to terminate capacity.

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

## Current B200 optimization reference

Read the [selected reference decision](../decisions/b200_monitor_score_reference.md)
and `experiments/b200_inference_benchmark/baseline.json` before preparing a new
comparison. The checksum-bound selection supersedes the October 6 GDN-only
reference below. Use the repaired two-logit score endpoint with cached causal
LAST pooling and the recorded native Gigatoken/direct FROST host recipe,
FP4 MLP/GDN/full-attention projections, selected SwiGLU/direct FP4 activation
output and MXFP8 full-attention prefill. Recurrence, KV cache and decode remain
BF16, with FP32 gates/state. Reuse archived controls and persistent caches;
compatible trials keep the candidate warm. Compare c1 latency and c128 prompt
throughput using all available matched baseline repeats; other combined-stack
concurrency controls are unmeasured. Preserve strict numerical failures and
explicit finite acceptance separately from passing checks. The acceptance is
specific to this adapter and pinned recipe, does not change training or frozen
past evaluations, and does not waive finite-output or future adapter checks.

Keep current selections in the checksum-bound baseline and its linked decision;
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
