# B200 large-prefill graph capture

2026-10-07. Piecewise graph replay works on the selected Direct FP4 serving
stack, but this trial regressed warm throughput and interactive latency. Keep
the selected reference and default recipe unchanged. Retain the implementation
as a named diagnostic path.

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
gain; counting fewer launches is insufficient. No further GPU trial followed
this negative result.

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

Retain the named graph-trial server API PID 98799 / engine PID 98881 on the
existing NC2 B200 pod `i243nsg10usytq`, port 8010. Health was 200, sole GPU
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
