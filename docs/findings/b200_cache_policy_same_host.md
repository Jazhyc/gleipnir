# B200 cache policy and restart comparison

2026-10-07. **Most of the NC2-to-EU latency reduction persists with the standard
cached recipe.** On the same EU-RO-1 B200, fresh cache-free serving reduces c1
median latency by 15.3%, with roughly 3% lower observed batch throughput and a
large persistent-cache memory saving. Restarting the already-fast cache-free
worker gives a much smaller latency change. The specific NC2 host/driver cause
remains unidentified; this does not establish a general fivefold cache-removal
speedup or an intrinsic slowdown from long process uptime.

## Protocol

Hypothesis: the EU cache-free latency reduction may reflect host/driver behavior
rather than cache removal or process age. Run `python -m experiments.b200_attention_gdn_serving.cache_policy_compare --name unique`
using the unchanged deployed snapshot so the concurrent repository refactor is
not an additional treatment. Measure resident cache-free, fresh standard cached,
then fresh cache-free on the same B200/driver, one server at a time. Reuse merged
weights and shared caches; retain native receipts and a bounded startup scoring
canary. Exclude whole-cohort warmup at each concurrency; collect three quick64/c1
and six full320/c128 repeats per phase, input tokens/s, latency bins, process CPU
and scheduler counters, paired scores and pooled/source AUROC against both the
same-host control and all archived selected repeats. Keep the final cache-free
worker warm. Stop on identity/input drift, failed score canary, nonfinite outputs,
OOM or timeout. This frozen training-seen diagnostic does not promote precision
or use the final ID set; chunked versus whole-prompt admission remains part of
the policy treatment. Current and restarted cache-free results separately test
resident-process effects. No profiler timing enters speed claims.


The three phases retain GPU UUID
`GPU-a2b24934-cf31-dd9e-91b5-397e49513463`, driver 580.178.04, the merged adapter,
native Gigatoken/direct FROST bindings, FP4 MLP/projections and causal MXFP8
attention. The policy treatment also includes the zero-storage metadata backend and
PIECEWISE graph configuration versus the standard mixed graph configuration;
it does not isolate an individual cache write or allocation. Fresh versus
resident process is tested separately. All GPU sources remain the validated deployed snapshot; only
the client is transferred from the concurrently refactored local repository.
Native receipts are reused. Fresh servers pass the bounded master/serving score
canary. Timing uses localhost HTTP, including serving work and excluding the
external network round trip. Four extra full320 passes reuse the final warm
worker to check throughput variation, without reloading or repeating startup.

## Results

| Phase | c1 p50 | c1 p95 | c128 input tokens/s |
| --- | ---: | ---: | ---: |
| Resident cache-free | 27.34 ms | 116.54 ms | 208,645 |
| Fresh standard cached | 30.36 ms | 116.11 ms | 211,041 |
| Fresh cache-free | 25.71 ms | 115.07 ms | 203,388 |

Use all three c1/six c128 repeats per phase. Fresh cache-free median latency is
**15.31% lower** than fresh cached; c1 input-token throughput is 10.11% higher.
p95 changes by less than 1%. Initial c128 throughput is 3.63% lower. The extra
four warm passes give **204,882 input tokens/s**, 2.92% below the same six cached
controls. Rates span 167,054–212,697 cached, 195,488–209,689 initial cache-free,
and 168,043–205,244 in confirmation. These overlapping ranges and the sequential
phase order limit a precise steady-state regression claim; do not cherry-pick
fast repeats or discard the late low-throughput passes.

Resident-to-fresh cache-free latency falls **5.95%** (1.63 ms), while initial
batch throughput falls 2.52%. This small change can include run-order/host
variation and does not prove an accumulation-related slowdown. The
fresh cached control already falls from the archived NC2 **149.69 ms** to
**30.36 ms**. The earlier fresh NC2 cache-free process measured 133.42 ms.
Thus the large historical contrast is predominantly associated with the host
and runtime environment, rather than cache policy or restarting this process.
CPU/NUMA placement, driver behavior, I/O or pod-level state remain hypotheses;
this experiment cannot distinguish them retrospectively.

No profiling is active during timing. API CPU per quick64 pass is approximately
0.13–0.15 s; engine aggregate CPU is 3.70–3.95 s, summed across its threads.
Main-thread scheduler wait is below 1 ms per c1 pass. These counters do not show
large frontend CPU saturation or descheduling in the current measurements, and
cannot diagnose the retired NC2 worker.

All c1 scores/margins are exact across policies and versus the selected archive.
Resident/fresh cache-free batch repeat-median scores are identical. Matched
cache-free versus cached c128 AUROC changes **+0.04298 pooled / +0.00782
source-macro percentage points**, with **zero threshold flips**. Only
`tool_trajectory/bash_arena` changes defined source AUROC (+0.22676 points);
all other defined sources are unchanged and the single-label Nemotron source
remains undefined. Mean/max absolute score differences are 0.000993/0.056559.
Calibration, partial AUROC, ties and all repeat diagnostics remain in the
summaries. Against all six archived NC2 repeats, fresh cached changes pooled
AUROC −0.07423 points with unchanged macro; cache-free changes pooled −0.03125
points and macro +0.00782 points. The warm confirmation has the same matched
ranking deltas. This is training-seen diagnostic evidence, not held-out quality
promotion. The checksum-bound selected baseline is unchanged.

Observed GPU process memory after the phase is **169,384 MiB cached** versus
**16,448 MiB cache-free**. Final RPC verifies zero runner/layer cache bytes and
no cache specifications. The remaining allocation includes weights, transient
within-prompt computation and allocator reservations; this is not a continuous
peak-memory measurement.

## Failures, startup and retained state

Preserve `cache_policy_euro01`: its persistent HTTP pool raises `ReadError` after
three c1 and four completed c128 passes. The server remains healthy with an empty
queue; no GPU stall is observed. `cache_policy_euro02` uses fresh pools per pass,
matching the archived wrapper client. It completes all three phases and the
additional four warm confirmation passes without transport failure or stall.

Cached control readiness takes 304.63 s with a first-use graph key; fresh
cache-free takes 119.07 s. Torch/Triton mutation analysis emits conservative
"assuming every input is mutated" warnings during cached compilation; compilation
continues and score parity passes. Preserve these warnings. No compiler or kernel
settings are changed to suppress them, and startup timings enter no speed claim.

Artifacts are `results/b200_attention_gdn_serving/cache_policy_euro02/`,
`cache_policy_euro02_warm_confirmation/`, both `cache_policy_euro02_*_start/`
receipts, failed `cache_policy_euro01/` and `cache_policy_collection01/`.
Executed clients, source bindings, all predictions, failed receipts, logs and
final worker state are collected locally (216 verified evidence files with a
checksum manifest, including the cache-archive receipt). The comparison
client's three focused tests and Ruff pass; commit `d34867c` contains only the
client/config/tests, preserving the concurrent refactor.

API **15761** / engine **15786** remain healthy and warm on port 8010, using the
experimental cache-free recipe. Shared compiler caches are preserved, including
new entries from both policies, additionally collected as
`.cache/runtime-resume/b200-serving-caches-euro-cache-policy-20261007.tar.zst`
(SHA256 `5ed67762c53b7b682d33daf892f9b8fa3abc5e51c6971c28828cc68014ee6067`). No capacity is created or terminated and no
serving or training default is promoted. Existing NC2 stall evidence remains
unresolved separately.

## User-selected cached restoration

The user retains the standard cached path: the roughly 5-ms median saving is
not valuable enough to trade away observed batch throughput. Prefix caching
was explicitly **disabled in both conditions**; it explains none of the gain.
The standard KV/state cache supports chunked prefill within a request. Our
cache-free implementation requires whole prompts, so the scheduler can leave
unused token budget when the next prompt does not fit. Changed batch sizes and
GEMM shapes are a plausible throughput cost, alongside the differing metadata
and graph policy, but have not been isolated by an operator/packing ablation.
This is a prototype/configuration tradeoff, not evidence that cache-free
attention is intrinsically slower. Preserve it as an optional memory-saving
path rather than the active serving reference.

`cached_restore_euro01` retires API/engine 15761/15786 before restoring the
unchanged selected cached recipe with native Gigatoken/direct FROST, persistent
compiler caches, chunked prefill and prefix caching off. Readiness takes
**264.96 s**; no throughput control is rerun. The startup score canary passes
existing **accepted-baseline agreement** (mean absolute difference 0.000807,
correlation 0.999967, nonzero adapter effect 0.839906). Strict master agreement
fails at mean absolute difference 0.020541, so `passed=false` and
`evaluation_passed=true` are preserved separately; existing finite acceptance
is reused, not replaced by a newly claimed strict pass.

API **16386** / engine **16409** remain healthy and warm on port 8010. Current
`cache_policy_state.json` records `standard_cached`; earlier cache-free closure
receipts above are historical. Restoration receipts, executed sources, retirement
and startup logs are collected under `cached_restore_euro01/` and
`cached_restore_euro01_collection/`. No kernel arithmetic, quantization, model
weights, cache policy flags beyond restoring the selected recipe, or capacity
lifecycle changes are made.
