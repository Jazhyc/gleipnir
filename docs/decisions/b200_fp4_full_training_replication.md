# Full regular 4B replication with native FP4 MLPs

Decision date: 2026-10-05. The user requests a complete training run followed by
ID evaluation and selects the original regular 4B run: 8,688 monitoring rows,
LR 5e-5, one epoch. This follows the explicit selection of native FP4 MLPs as
the B200 execution default. Quality evaluation is pending; this decision does
not establish equivalence with BF16.

## Frozen comparison

Use `experiments/b200_fp4_full_training/config.yaml` and the named
`qwen35_4b_b200_fp4_mlp` profile. Preserve the original regular prompt, Kimi soft
targets, seed 0, initial FP32 master adapter, rank 128/alpha 256, AdamW schedule,
logical batch 32, packing budget 16,384, no checkpointing or truncation, and
final one-epoch selection. Expect 272 updates and 83,816,369 input tokens.
Frozen source checksums, lineage holdouts and the executed source archive bind
the run to the original inputs; no prompts or target distributions change.

Reuse the original regular control and its cached ID scores. Its training used
BF16 MLPs and SDPA; the candidate combines native FP4 MLPs and BF16 FA4. Report
their differences as a combined recipe comparison, not an isolated FP4 effect.
Historical startup/cache conditions also differ. No additional full BF16
control, ID tuning, alternate epoch selection, OOD evaluation, publication or
quality promotion is part of this request.

Evaluate all 3,012 canonical CoT-removed ID examples using the regular prompt.
The final FP32 master adapter is rebased for BF16 serving. Require a fresh,
bounded master-to-vLLM score-parity receipt and a nonzero adapter effect before
scaling evaluation. Report ranking metrics, calibration, thresholds, ties and
per-source results against the exact same cached control population. One seed
does not establish equivalence.

## Execution and retained worker

Use the existing US-NC-2 B200 pod `i243nsg10usytq`, network volume `ixbh81vf9c`,
and populated shared compiler/kernel caches. No capacity is created or
terminated. The old resident timing worker, PID 11905, supports only twenty-step
trials; preserve its state/logs and replace that process with the ordinary full
Trainer through a retention wrapper. Pipeline PID 16069 launches trainer PID
16214. After training, keep the model, native plans and initial CPU FP32 master
copies resident, release optimizer state, and park the worker for future work.

Reuse checksum-bound startup validation from
`results/b200_mlp_gemm/warmed03/causal_adapter/training_metadata.json`, SHA256
`14ab15279bb8895cf32353117d5c1cf957ad45d2b0ca9d7205067db27d77edeb`.
Retain historical failed strict diagnostics and explicit selected-finite
acceptance. Continue finite/missing-gradient checks on every update. No model
warmup replay, packing canary or longest-batch numerical preflight is repeated.
Adapter-specific serving parity remains required.

Artifacts live in `results/b200_fp4_full_training/`; logs in
`logs/runpod/b200_fp4_full_training/`; input manifest in
`data/b200_fp4_full_training/manifest.json`. `execution_contract.json` binds
executed sources, command and manifest. `worker_replacement.json`, `launch.json`
and `worker.json` record process replacement and current state. No in-chat
scheduling tool is available; active-turn checks cannot promise agent follow-ups
after the turn ends.

## Full-corpus cache costs

The original epoch contains 4,550 physical batches with 3,086 unique packed token
counts. Native plans are keyed by exact token count, so the short warmed timing
cohort does not describe first-use overhead across this population. The first
candidate update takes about 397 seconds including compilation; early subsequent
updates take roughly 85–113 seconds. These are interim wall times, not completed
epoch throughput or the previously reported warmed 3.678-second cohort result.

A bounded compile-only probe shows persistent-object cache hits and a remaining
first-use miss. Use an optional 16-process helper to populate the unchanged
native plans in spare CPUs, without loading or replaying a model or executing
GEMMs. Preserve shared cache namespaces and atomic upstream cache publication.
Archive helper source, future shapes, cache hit/miss counts and execution time
separately; account for this concurrent work when reporting full-run timing.
The first parallel pilot incorrectly rejects the backend's one-byte workspace
marker. Preserve that failure; the corrected helper permits only that marker,
requires zero fused workspace, and checks GPU headroom for compiler contexts.

Full training and ID results remain pending. Collect and verify final artifacts
before drawing quality or practical full-corpus performance conclusions.

## Full-corpus throughput regression and runtime-shape fix

The first attempt was paused after 43 of 272 updates with its live state intact.
The latest completed update takes about 45 seconds, after earlier updates of
roughly 80–113 seconds; this is substantially slower than the historical control
(3,916 seconds for the whole epoch). The user expects roughly an hour. Do not
extrapolate the 3.678-second short-cohort measurement to this token distribution
or accept a multi-hour run without investigating the regression.

A compile-only CPU profile of the fourteen packed shapes at historical update
44 takes 23.980 seconds to construct 56 native plans, despite 168 compiled-object
cache hits and zero misses. Repeated graph construction, module loading, template
reads and runtime-library discovery dominate this bounded diagnostic. This is
not a full-training trace, and does not establish exact end-to-end percentages.
Another bounded probe takes 7.275 seconds to compile/load FP4 conversion and
row-scaling variants for the same shapes. The earlier compile-ahead helper
populated GEMM objects, not these Triton variants. Its successful 747-second run
therefore did not resolve the underlying per-shape overhead.

NVIDIA's existing compiled GEMM already supports symbolic runtime M. Reuse one
fused-descaling plan per device/K/N geometry, retaining all geometry, scale-blob
and native launch guards. Keep the fixed-M direct API for historical probes.
Pass conversion group counts and row-scale lengths at runtime instead of
specializing on each count. Preserve K, hardware conversion, fused BF16 rounding,
tile configuration, FP32 adapter masters and all experiment controls.

The bounded native proof covers four projection orientations at M=129, 5,047,
16,322 and 29,337: all sixteen outputs are bitwise identical to fixed-M plans,
with the same NVIDIA tile configuration and no refused launches. A separate
conversion proof covers all three activation widths at seven row counts,
including empty-valued rows and 128-row boundaries: packed codes, scale bytes
and FP32 inverses match bitwise in all 21 cases. Each width uses one compiled
conversion kernel across these lengths. Generic row scaling matches bitwise
at the same seven lengths with one compiled variant. Record these receipts and
source hashes separately from the reused historical startup receipt; unchanged
model canaries are not claimed to have passed again. Full-run throughput and
ID quality remain unmeasured for this fix.

Applying the Python cache-key change requires a process restart. The first
attempt has no intermediate disk checkpoint; a replacement must restart from
the frozen original initialization, preserving the slow attempt's source,
logs and diagnostic receipts. Reuse the same disk caches and allocated B200.

The replacement pipeline PID 66650 and trainer PID 66778 start on the existing
B200 at Unix time 1791235348.779. The slow attempt and logs are archived under
`results/b200_fp4_full_training_slow_attempt01/` and
`logs/runpod/b200_fp4_full_training_slow_attempt01/`. Its 43 updates have no
resumable checkpoint and are not counted toward the replacement epoch.
`runtime_shape_validation.json` binds the 16 GEMM, 21 conversion and seven
row-scale cases plus current source checksums; the new execution contract
records that receipt's hash. Sixty-one focused tests and Ruff pass.

That replacement stops before any update: the original startup source guard
correctly rejects the changed MLP source. Preserve this failure under
`b200_fp4_full_training_source_guard_attempt02` results/log paths. The source
binding now retains historical hashes separately and pins the new hashes to
the 44-case bitwise proof; startup metadata records that separate equivalence
evidence without relabeling old canaries. The same source check runs before
model loading in the campaign launcher. A bounded receipt read passes with
no repeated model arithmetic, and 62 focused tests pass. The corrected pipeline
PID 67204 and trainer PID 67332 restart from the same frozen controls at Unix
time 1791236020.868; throughput and ID remain pending.
The per-shape compile-ahead helper is unnecessary for the runtime-M baseline.
