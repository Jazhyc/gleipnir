# Small production inference benchmark

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
