# Selected reference concurrency scaling

Measure the repaired selected FCFS scorer at client concurrency 1, 2, 4, 16,
32, 64 and 128 on the existing verified B200. Keep the same frozen full320
prompts, order, 1,310,581 input tokens and labels at every level. This avoids
confounding client concurrency with the earlier quick64/full320 cohort switch.
This is a training-seen systems cohort, not a production arrival distribution
or held-out model-quality evaluation. No candidate or precision promotion.

The user explicitly requested a fresh sweep including c1/c128 on 2026-10-07.
Exclude one full workload warmup at each level, then measure three timed repeats
with fresh HTTP pools. Preserve all per-request scores and times. Report median
and repeat range for prompt tokens/s, requests/s, p50/p95/p99 response latency
and length-bin latency. These are closed-loop results; semaphore wait before
HTTP dispatch is excluded from request latency. Keep model/adapter, native
Gigatoken/direct FROST, FP4/MXFP8 recipe, 32768 token budget, 128 engine sequences,
cache policy, causal LAST pooling and exact two-row classifier fixed.

Use the selected adapter canary and native receipts before timing. Compare
AUROC, calibration, score drift, ties and flips against every saved c128/full320
reference repeat at all levels. This is a quality/batch-shape diagnostic;
archived c128 timing is not a matched lower-concurrency timing control. Stop on
provenance/input drift, canary/finite/token-count failure, server failure/OOM or
suite completion. Keep the explicitly requested reference server warm afterward;
do not alter capacity or remove shared compiler/kernel caches.

```bash
PYTHONPATH=src:. python -m experiments.b200_score_scaling.run \
  --name scale01 --retired-parent <selected_server.json> \
  --source-bootstrap-dir <receipt-bound-frozen-source-bootstrap>
```

The bootstrap is needed when current package moves invalidate the old reference
source bindings; preserve the current checkout and rehydrate the exact recorded
serving sources before launching. Reuse the checked snapshots from
`results/b200_length_admission/reference_sources_applied01/` and the matching
project/runtime metadata. The bootstrap retains pinned CUTLASS import selection.
No source or dependency checks are waived. Shared compilation identity remains
`f0290e9cc3`; scheduler admission stays upstream FCFS. Results and executed source
receipts live under `results/b200_score_scaling/<name>/`.

The completed sweep plateaus near client concurrency 16; see
[the finding](../../docs/findings/b200_score_scaling.md) for medians, repeat ranges
and quality variation. Export its table and standalone figures with
`python -m experiments.b200_score_scaling.summarize results/b200_score_scaling/scale01`.

## Current FP8 recipe on NC2

The user requests a fresh host scaling comparison on 2026-10-08. Hypothesis:
the current FP8/vLLM 0.31 scorer reaches a similar throughput plateau and latency
curve on the new NC2 B200. `current_config.json` freezes the same full320 order
and 1,310,581 tokens at c1/2/4/8/16/32/64/128, one excluded full warmup per level
and three timed repeats. Keep the current serving selection and 128-sequence/32K
engine limits fixed; reproduce its adapter canary before timing. Compare quality
with all six archived current-FP8 c128 repeats, retaining calibration, ties and
undefined sources. This training-seen cohort is diagnostic, with no promotion.
The old full scaling curve used 0.24/FP4 attention projections; label that recipe
difference and use the same-recipe archived FP8 c128 point separately. Stop on
identity, canary, finite/token-count, native/server/OOM failure or completion.
Retain a successful scorer warm and preserve caches. Run with the staged 0.31
interpreter through `experiments.b200_vllm031.runtime -m
experiments.b200_score_scaling.current --name <run> --expected-gpu-uuid <live-uuid>`.
Export the completed sweep's medians, ranges, quality deltas and standalone
PNG/SVG curve with `python -m experiments.b200_score_scaling.current_summary
results/b200_score_scaling/<run>`. The linked finding records the NC2 result and
distinguishes the old full curve from the matched-recipe c128 comparison.
