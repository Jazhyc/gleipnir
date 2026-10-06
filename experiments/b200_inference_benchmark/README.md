# Small production inference benchmark

Current B200 optimization baseline, selected by the user on 2026-10-06:
**FROST FP4 MLPs and large GDN QKV/Z/output projections plus cuDNN MXFP8
full-attention prefill**, with BF16 small gate projections, convolution/recurrence,
KV cache and decode, plus FP32 gates/state.
`results/b200_attention_gdn_serving/fp4_gdn_cudnn_mxfp8_02` is the archived
control; `mxfp8_confirmation02` holds five warm c128 passes (173,938 input tokens/s).
`baseline.json` binds the workload, executed recipe, result and explicit
`user_accepted_finite` quality receipt. The user accepts macro-AUROC +1.68 and
pooled AUROC −0.37 percentage points relative to the preceding FP4 reference;
strict score/native-precision failures remain intact. New kernel
conditions use `baseline: selected`; explicit historical references remain
unchanged. Primary comparison is c128 throughput, retaining peak, latency and
ranking diagnostics. Full-cohort candidates use `high_reference: selected` too,
which binds the archived full-sweep checksum. See the
[baseline decision](../../docs/decisions/b200_mxfp8_inference_baseline.md).

Hypothesis: a fixed, small real-prompt workload can expose latency and throughput
tradeoffs before changing the monitor's production serving kernels or precision.
Establish a BF16 dynamic-LoRA baseline first; no optimization sweep is launched.

Reuse the exact 320 identities and prompts from the training systems cohort.
The default quick pass selects 64 evenly spaced prompt-length ranks, including
the shortest and longest, then uses a fixed seed-0 shuffled order. The full
320-row mode uses the same seed and parent population. This is a systems
development set containing training examples, not a held-out quality evaluation
or a representative sample of an established production traffic distribution.
Record its source composition and lengths; labels never select an optimization.

Use the final FP4-trained 4B adapter with the pinned unquantized BF16 base and
BF16 serving. Training FP4 kernels are not enabled at inference. Keep one vLLM
0.24.0 HTTP server on the existing NC2 B200 across the benchmark and subsequent
compatible trials. Reuse the shared serving/compiler caches. The first baseline
uses FlashInfer GDN, 16 engine sequences, 32,768 scheduled tokens, 32,768 context
and 25% GPU memory utilization. Bind only localhost; no public service is deployed.

Measure two passes at each client concurrency 1, 4 and 16. Each request submits
the rendered prompt as text, so HTTP serialization, server tokenization, queueing
and model inference are included in request latency. This is a closed-loop load
test: a client starts its next request when its previous request finishes.
It does not estimate an open-loop arrival-rate SLO. Output is one constrained
0/1 token. Report completed requests/s, prompt tokens/s, p50/p95/p99 response
latency and latency by prompt-length bin; retain every score, logprob and timing.
For a single-token response this measures decision latency, not decode throughput.

Disable prefix caching for the baseline so identical replay passes cannot skip
prefill. Future prefix-cache tests must define their own realistic hit rates.
Initialization, score-parity checks and a four-length warmup are separately timed.
The installed HTTP API lacks explicit `logprob_token_ids`; request the two
allowed decision tokens and processed top-two logprobs, identified by token ID.
Their normalized probability is the same binary margin when only the allowed
token mask is applied. Verify this API path against the archived twenty-row
master/serving canary, including a nonzero adapter effect, before timing.

Stop on provenance drift, failed canary, missing/nonfinite decision logprobs,
token-count mismatch, truncation, server failure or OOM. Preserve failed outputs.
Compare future candidates with matched per-row baseline scores and retain
within-baseline variation; no numerical acceptance or production deployment is
authorized by a speed result alone. Finish after the baseline suite, collect its
artifacts and keep the server alive for future authorized inference work.

```bash
python -m experiments.b200_inference_benchmark.run --prepare-only
python -m experiments.b200_inference_benchmark.run --output baseline01
python -m experiments.b200_inference_benchmark.run --rows 320 --output full01
```

Use `--reuse-server` for compatible repeated client trials. It verifies the
frozen rendered-input checksums and configuration, avoiding rereading and
retokenizing the raw source population. Artifacts live under
`data/b200_inference_benchmark/` and `results/b200_inference_benchmark/`; logs
under `logs/runpod/b200_inference_benchmark/`. The server PID and command are
recorded. No capacity is created or terminated. No in-chat scheduling tool is
available: active-turn startup checks do not promise monitoring after the turn.

## Completed baseline, 2026-10-06

All six timed passes complete on the 64-row quick workload (269,411 prompt
tokens, 188–28,733 tokens per request). Medians across the two passes:

| Client concurrency | Pass seconds | Prompt tokens/s | Requests/s | p50 latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 14.138 | 19,074 | 4.531 | 0.154 s | 0.639 s |
| 4 | 7.639 | 35,269 | 8.378 | 0.411 s | 0.872 s |
| 16 | 5.602 | 48,096 | 11.425 | 1.255 s | 2.143 s |

Following the user's explicit reporting preference, lead optimization comparisons
with prompt-token throughput, alongside requests/s and latency. Each request
generates only one decision token; output-token throughput consequently tracks
requests/s. Keep the same prompt-length mix and cache policy across candidates.

These are localhost closed-loop results with prefix caching off, rather than
production arrival-rate/SLO estimates. The cohort is shorter on average than
the ID population. Across the six score arrays, mean/max per-row range is
0.008312/0.062177 and three rows cross 0.5; later candidates must account for
that baseline variation. Raw outputs and length-bin latency tails are retained.

HTTP parity passes against the archived master reference: adapter mean absolute
score difference 0.003817 and correlation 0.999770; nonzero adapter effect 0.782080.
The successful server readiness wait is 772.006 seconds, parity 18.235 seconds
and four-length warmup 0.818 seconds, separate from timed requests. Earlier
preparation/PATH failures remain archived. Server PID 71421 and engine PID 71522
remain resident on pod `i243nsg10usytq`, localhost port 8000, using about 48.8 GB
GPU memory. Eight focused benchmark tests and Ruff pass.

## Warm reuse and throughput bounds

The retained server is useful for repeated requests, load/concurrency passes and
profiling the same implementation. Changing loaded kernel backends, precision,
engine sequence/token budgets, memory allocation or compilation/CUDA-graph
settings normally requires a new vLLM process. Preserve disk caches across that
restart; compatible cached builds can be reused, while model loading, runtime
plans and graph capture still have process startup costs. No restart or new
concurrency sweep is performed for this analysis.

Concurrency 16 is the current `max_num_seqs` limit, not a demonstrated optimum.
Client load beyond 16 cannot raise the scheduled sequence limit; it can keep the
queue fed but primarily adds queueing. The current cache planner estimates
975,592 token slots, or 29.77 full 32,768-token contexts. This is a cache-capacity
estimate, not supported active concurrency or a throughput measurement. Increasing
engine concurrency changes runtime buffer/graph allocation too. Throughput rose
35,269 → 48,096 prompt tokens/s between tested loads 4 and 16; saturation of a
larger engine remains unmeasured.

A loose arithmetic-only ceiling for this frozen quick workload is approximately
263,000 prompt tokens/s. Count all BF16 decoder projection and LoRA affine work
(7.478 GFLOPs/token), plus causal full-attention QK/PV work at the actual lengths
(1.080 GFLOPs/token). Divide a nominal 2.25 PFLOPS dense BF16 B200 peak by that
8.557 GFLOPs/token total. NVIDIA's [HGX specifications](https://www.nvidia.com/en-us/data-center/hgx/)
list 36 sparse BF16 PFLOPS for eight B200s, with dense performance half sparse.
The estimate ignores GDN scan/solve/convolution/normalization, memory traffic,
kernel efficiency, dispatch/HTTP overhead and power/clocks. It is an optimistic
hardware bound, not a prediction for the current kernels or a claim that a
fivefold improvement is attainable. The practical software ceiling is unknown.
Record the assumptions in `arithmetic_ceiling_estimate.json`.

## Merged BF16 benchmark protocol

The user selects merged LoRA serving for future evaluations and requests this
next benchmark. Hypothesis: removing adapter projections/dispatch improves
throughput and latency. Freeze the same 64 rows, order, engine sequence/token
budgets, memory fraction, backends, prefix-cache policy and six timed passes.
Compare with the completed dynamic-LoRA baseline; do not rerun full ID or tune
the serving recipe on labels.

Materialize the pinned unquantized base plus final FP32 master LoRA updates on
CPU, accumulating each `W + (alpha/r) B@A` in FP32 and rounding once to BF16.
Check every merged weight finite, every adapter pair represented and a nonzero
weight effect. Copy unchanged tensors and tokenizer/config assets. Keep source
weights immutable and bind shard/adapter/config hashes in a persistent receipt.
The generated checkpoint lives only at
`/tmp/gleipnir-merged/fp4-full-training-bf16` on the Runpod container disk; it is
disposable and reconstructable. No duplicate merged checkpoint goes on the
network volume. Unsupported adapter variants fail rather than silently merging.

The merged server omits all dynamic-LoRA flags and uses the same localhost
endpoint after retiring the idle dynamic server, whose command/logs stay archived.
Reuse shared compiler caches. The fresh twenty-row serving canary compares with
the archived FP32 master and dynamic-LoRA scores, requiring the unchanged
0.02 mean score difference/0.99 correlation bounds and a nonzero effect against
the archived unadapted base. Stop on failed parity or structural/finite checks.
Report all paired 64-row score/margin differences and threshold flips against
each baseline concurrency's repeat median, including its own baseline variation.
Keep the merged worker warm if the benchmark passes.

```bash
python -m experiments.b200_inference_benchmark.merge_model
python -m experiments.b200_inference_benchmark.run --output merged01 \
  --merged-model /tmp/gleipnir-merged/fp4-full-training-bf16
```

## Completed merged benchmark

All six matched passes complete in `results/b200_inference_benchmark/merged01`.
Each cell below is the median of two repeats on the same 64 prompts (269,411
input tokens), with prefix caching disabled. Startup is excluded.

| Client concurrency | Dynamic LoRA input tokens/s | Merged input tokens/s | Ratio | Merged p50 / p95 latency (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 19,074 | 29,763 | 1.56× | 0.104 / 0.360 |
| 4 | 35,269 | 69,049 | 1.96× | 0.224 / 0.435 |
| 16 | 48,096 | 107,073 | 2.23× | 0.563 / 0.946 |

Median requests/s are 7.070 / 16.403 / 25.436 and pass durations
9.056 / 3.902 / 2.516 seconds. Interactive p50 falls from 0.154 to 0.104
seconds; at concurrency 16, p50/p95 fall from 1.255/2.143 to 0.563/0.946.
This is the matched serving effect of merging, including the graph/kernel path
changes caused by disabling dynamic LoRA, not a full ID or production SLO result.

Fresh twenty-row parity passes against the archived master: mean absolute score
difference 0.00165113, correlation 0.99988104; against dynamic LoRA:
0.00237397 and 0.99990730. Maximum adapter effect against reused baseline base
scores is 0.77222942. On 64 paired repeat medians, score mean/max differences
at concurrency 1/4/16 are 0.004552/0.049130, 0.004371/0.049130 and
0.004815/0.062419; margin mean/max differences are 0.036133/0.25,
0.032227/0.25 and 0.036133/0.25. There are 1/1/2 threshold flips at 0.5.
The merged model's six-pass mean/max score range is 0.000457/0.027824,
with zero threshold-unstable rows, versus baseline 0.008312/0.062177 and three.
Preserve these finite score shifts; passing canary does not establish identical
held-out metrics or justify attributing differences solely to BF16 merge rounding.

The CPU merge takes 76.953 seconds, changes all 128 projection weights and uses
about 8.8 GiB ephemeral disk. Server readiness takes 700.771 seconds, parity
10.789 and four-length warmup 0.472, separate from timed passes. Package imports,
compilation and runtime setup dominate readiness; checkpoint shard loading takes
1.61 seconds. Source FP32 master and rebased adapter checksums are unchanged.
Thirteen focused tests and Ruff pass. Results, logs, executed sources and merge
receipts are collected; disposable weights stay on the pod only.

Merged server PID 72552 and engine PID 72648 remain resident and healthy on
localhost port 8000, about 49.8 GB GPU memory. For compatible request trials:

```bash
python -m experiments.b200_inference_benchmark.run --output merged02 \
  --merged-model /tmp/gleipnir-merged/fp4-full-training-bf16 --reuse-server
```
