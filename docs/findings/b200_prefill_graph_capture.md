# B200 large-prefill graph capture

2026-10-07. Larger piecewise graphs work on both the historical NC2 generation
stack and the repaired EU scoring stack, but neither trial improves batch
throughput. Keep the [repaired score reference](../decisions/b200_monitor_score_reference.md)
selected. Retain larger graphs as a named diagnostic path.

## Repaired EU score endpoint retry

The user selected `mutation03` as the reference before this trial. Preserve its
cached causal LAST pooling, exact two-row BF16 classifier, FP4 backbone,
MXFP8 prefill, native Gigatoken/direct FROST bindings and validated mutation
analysis. `experiments/b200_score_graphs` combines the existing graph and score
worker load hooks; each audit runs once and the backbone loads once. Reuse the
same thirteen capture buckets up to 32768, padding/producer-band guards and
temporary profiling-cache fix described below. No new GPU capacity or timing
control: compare all saved same-host three c1/six c128 repeats.

`graphs01` becomes ready in 248.68 seconds, including 180.48 seconds compiling
its new configuration key `a900a86820`. Capture finishes in five seconds and
uses 3.79 GiB; runtime cache capacity remains 4616812 tokens. Shared caches are
retained. No mutation-analysis warnings appear. Twenty-four changed token-ID
classifier fixtures cover boundary padding, fallback, large replay and bypass.
Score/logit/repeat differences are exactly zero. The adapter canary passes
against the accepted cached artifact, with mean error 0.000807, correlation
0.999967 and nonzero effect; its inherited strict master failure remains explicit.

Use the frozen quick64/full320 text workloads and fresh HTTP pools. Exclude one
warmup before each concurrency's three/six timed repeats. All measurements are
end-to-end localhost requests, including encoding; they exclude external network
latency. The full cohort is training-seen systems development, not final ID.

| Warm median metric | Repaired reference | Larger graphs | Change |
| --- | ---: | ---: | ---: |
| c1 request p50 | 31.225 ms | 30.627 ms | -1.92% |
| c1 request p95 | 128.537 ms | 128.502 ms | -0.03% |
| c1 input tokens/s | 97272 | 99069 | +1.85% |
| c128 input tokens/s | 215707 | 210494 | -2.42% |
| c128 request p50 | 2260.057 ms | 2298.744 ms | +1.71% |
| c128 request p95 | 2615.344 ms | 2794.006 ms | +6.83% |
| c128 source-macro AUROC | 0.873931 | 0.873472 | -0.04595 pp |
| c128 pooled AUROC | 0.887170 | 0.886857 | -0.03125 pp |

Candidate c128 rates span 203006--218993, versus 212562--217083 for the saved
reference; sequential schedules and repeat variation limit causal attribution.
Median paired c128 score mean/max differences are 0.003204/0.076474, with zero
threshold flips. Changed source AUROC is insider trading -2.0 pp, soft trigger
+0.5540 pp and bash arena +0.1134 pp; other defined sources are unchanged.
Undefined single-label sources, calibration, ties and every repeat are retained
in `c128_comparison.json`. At c1, source-macro AUROC is unchanged and pooled
AUROC falls 0.14778 pp, mean/max score drift is 0.006508/0.195115, with zero
flips. Synthetic exactness does not imply exact real-prompt outputs with padding.

Two separate post-timing c128 profiles toggle replay in this same compiled
worker, with an excluded warmup for each. The bypass profile is a diagnostic,
not a rerun of the selected timing control. Replay has 1353 `cudaGraphLaunch`
calls and 15241 ordinary launches, versus 41193 ordinary launches in bypass:
**59.72% fewer total launch calls**. GPU kernels are 42266 versus 41193; graph
capture groups submission and does not remove GPU kernels. Kernel-busy unions
are 5.643/5.636 s, GPU windows 6.063/6.053 s and idle gaps 0.420/0.416 s for
replay/bypass. Launch grouping does not reduce the measured GPU gaps. Eager
attention/GDN and scheduler/transfer work remain outside captured segments;
these traces do not establish a particular host component as the causal bottleneck.

All 64 collected graph artifacts verify against the remote checksums. The
snapshot includes the executed frozen-source graph helper, whose SHA256 is
`aa394c2062256616a5fe89dce958e1fb7eea1a2d17e8d998c08accb07a3eac79`.
Graph closure is healthy with sole engine 18599, then the trial is retired.
Restore the selected small-graph scorer through `graphs_restore01`, using its
saved command and native repair receipt, retained caches, adapter canary and
excluded warmups only. The restoration becomes ready in 98.54 seconds, reuses the exact selected
`f0290e9cc3` compilation identity, passes its adapter canary, completes both
excluded warmups and records no timing-control repeats. Closure is healthy with
API 19168 / sole GPU engine 19191, 170840 MiB and zero mutation warnings. Its
35 artifacts are retained with checksum verification.
Forty-five focused tests passed before launch; 47 pass after adding profile and
restoration checks, with Ruff passing. Trial/profile/quality evidence lives in
`results/b200_score_graphs/graphs01/`; no graph promotion is made.

## Historical NC2 generation trial

## Intervention and validation

`experiments/b200_attention_gdn_serving/prefill_graphs.json` retains the selected
merged model, FP4 MLP/projections, combined preparation, tuned GEMMs, cuDNN
MXFP8 prefill, BF16 GDN and scoring/batching settings. It excludes the deferred
direct-GDN-output candidate. Set PIECEWISE mode and capture token sizes
1, 16, 64, 128, 256, 512, 1024, 1536, 2048, 4096, 8192, 16384, 32768. Preserve
the attention/GDN/KV-update splitting ops. Reject live graph padding above
12.5% or across FP4 producer boundaries at 1536/4096 rows.

Bind the installed dispatcher and model runner by checksum. The first native
startup (`prefill_graphs02_serving01`) failed before capture: vLLM 0.24's
`_init_minimal_kv_cache_for_profiling` used the maximum capture token count as
KV block count. Its supposedly minimal 32,768-block hybrid cache allocated
about 136 GiB before another 66-GiB allocation failed. Adapt only that temporary
initializer's `min_blocks` assignment to `max_num_seqs` (128). PIECEWISE dummy
profiling excludes real attention execution and needs request-sized state
storage. Keep runtime KV allocation and capture token sizes unchanged. The
generated initializer SHA256 is
`63e8ebf38a03ef10c353b991c772724d33b029add8b6f98ba2ad6696b675124b`.
An earlier client-alias correction and both retired attempts are preserved.

The successful server became ready in 640.59 seconds. Estimated graph memory
was 5.09 GiB; actual capture took four seconds and 5.04 GiB. Remaining KV
capacity was 4,529,927 tokens (138.24 full-length 32K requests), supporting the
unchanged c128 workload. Shared compiler/native caches were reused.

Compare 24 synthetic fixtures, two changed token values at 12 lengths
127--32760, with graphs bypassed in the same worker and then replayed twice.
Exercise padding, precision-boundary fallbacks and large replay. All outputs
were finite; maximum score/logprob/repeat differences were exactly zero. The
20-example adapter canary passed and matched the selected serving control
exactly, with a nonzero adapter effect. Existing native receipts and their
historical precision failures remain unchanged. All 46 live source hashes
matched; 36 focused tests and Ruff passed.

## Warm serving results

Use the unchanged frozen 320 training-seen systems-development examples,
1,310,581 input tokens, manifest SHA256
`b02af232d76935f2cda2a52213ebce27ae04ca5aaf44898cfe496104adad4266`.
Exclude final ID from selection. Reuse the archived selected control. Five
c128 repeats follow one excluded full warmup, then two c1 passes on the
frozen 64-example latency workload.

| Metric | Selected reference | Graph trial | Change |
| --- | ---: | ---: | ---: |
| Median c128 input tokens/s | 196,866 | 192,943 | -1.99% |
| Aggregate c128 input tokens/s | 196,857 | 192,066 | -2.43% |
| c1 median latency | 159.46 ms | 172.44 ms | +8.14% |
| c1 p95 latency | 275.92 ms | 314.92 ms | +14.13% |
| Source macro AUROC | 0.878105 | 0.878222 | +0.0117 percentage points |
| Pooled AUROC | 0.885842 | 0.885490 | -0.0352 percentage points |

Candidate warm rates: 195420, 195098, 188260, 192943, 188849 input tokens/s.
Median-score pairing found zero threshold flips, mean/max score differences
0.001269/0.097038. Candidate pooled Brier was 0.148187, balanced accuracy
0.777817, recall 0.753165 and FPR 0.197531; 84 distinct median scores. Full
per-source metrics, calibration and repeat variation are in the confirmation
receipt. Small development AUROC changes do not establish generalization or
a graph quality gain.

## Replay profile and interpretation

Profile a separate c128 pass on the same warmed server, outside benchmark
timing. Replay is verified: 1,287 `cudaGraphLaunch` calls. Individual launch
APIs fell from 44,784 to 18,949; including graph launches gives 20,236 calls,
a 54.8% reduction. Thirty-nine of 43 batches used PIECEWISE replay, including
38 exact 32K batches. Padding added only 100 tokens (0.0076%), so additional
padded arithmetic is not a plausible main cause of the c128 regression.

| Profile quantity | Saved reference | Graph trial |
| --- | ---: | ---: |
| GPU batches | 42 | 43 |
| GPU kernels | 44,784 | 45,852 |
| Sum of kernel durations | 5.8521 s | 5.7981 s |
| Union of kernel intervals | 5.8507 s | 5.7939 s |
| First-to-last kernel window | 6.4253 s | 6.4784 s |
| Gaps within that window | 0.5746 s | 0.6845 s |
| Kernel busy fraction in that window | 91.06% | 89.43% |

Graph capture groups launches; kernel count per batch is essentially unchanged.
Graph launch API duration had a 67.18-us median and 98.29-us p95. Summed launch
API spans fell only from about 0.276 s to 0.256 s because each graph submission
cost more than an individual launch. Recorded `cudaMemcpyAsync` spans rose
from 1.756 s to 3.856 s, with more event wait time. These spans can block on
GPU work and overlap CPU operator scopes; do not interpret them as additional
wall-time fractions or proof of more data-transfer work. `aten::copy_` CPU
duration similarly includes waiting. Different batch schedules and separate
profiled runs prevent a causal allocation of the regression.

Launch grouping worked, padding was minimal and summed kernel time was slightly
lower, without improved application throughput. Eager attention/GDN and
scheduler/state/transfer handling remain outside these graphs. Further tuning
needs to address replay cost or those boundaries and demonstrate a measured
gain; counting fewer launches is insufficient. The subsequent EU scoring retry is recorded above.

## Artifacts and retained state

Paths are under `results/b200_attention_gdn_serving/`:

- `prefill_graphs03_serving01`: sweep, canaries, executed sources;
  summary SHA256 `82d15aca368454904adead12800cbbc2a1659641517d918ced52144191d2626b`.
- `prefill_graphs03_confirmation01`: five warm throughput/quality repeats;
  summary SHA256 `46aa16c8aea3daa88320040282acfc6724f0a1fea536321811e621a6fae04b74`.
- `prefill_graphs03_latency_confirmation01`: two warm latency passes;
  summary SHA256 `d9e91cdc6f0ed2d0d1e8a0a60488e50a6c18b7afb9a42d1c85025f0773956dd5`.
- `prefill_graphs_profile01`: raw trace, breakdown, API comparison and
  before/after dispatch counters; trace SHA256
  `2b4581623bb98cca5ac65d5bd89313ff52d1e457ca83fcfe2cfb06749a3cb402`.
- `prefill_graphs_collection01`: runtime receipts, generated initializer,
  health, logs and retired startup receipts; all 21 collected file checksums
  and 12 profile checksums verified locally.

At collection, the named graph-trial server API PID 98799 / engine PID 98881 was on the
then-existing NC2 B200 pod `i243nsg10usytq`, port 8010; it was later retired. Health was 200, sole GPU
process, memory 169190 MiB, temperature 32 C at collection. Campaign state is
`completed_server_retained`, explicitly not promoted. Selected baseline
SHA256 remains `39811c43e0b4bbf574e682d7b21f09e394909af3af4a69f3b398193cace89166`.

Startup is unusually slow for this 4B model. Receipts show merged BF16 700.8 s,
selected Direct FP4 680.8 s, previous restart 668.5 s and this graph trial
640.6 s, rather than a monotonic slowdown from precision modifications. API
and spawned-engine imports each take roughly three minutes in the
network-backed environment; compilation took 188.2 s, weight loading 1.38 s
and capture four seconds. Earlier process inspection showed filesystem
request waits during imports. Import/compile-cache diagnosis is separate from
warm inference optimization. Extra elapsed time in this session includes the
retired alias correction and profiling-cache OOM attempts.
