# B200 inference kernel screen

Hypothesis: native vLLM per-channel-weight/per-token-activation FP8 MLP GEMMs
improve merged-model throughput and latency on B200. Use the pinned 0.24.0 online
quantizer and native kernel selector, rather than training autograd wrappers.
Weights are quantized once during loading; include activation conversion in all
end-to-end timings. Only the 32 decoder MLPs change precision. Full attention,
GDN projections/recurrence, embeddings and vocabulary head remain BF16.

Freeze the existing 64 prompts, 269,411 input tokens, shuffled order, one-token
HTTP scoring contract, seed 0, concurrency 1/4/16 and two repeats, prefix caching
off, 32,768 context/token budget, 16 engine sequences and 0.25 memory fraction.
Compare with the completed merged BF16 suite, not the unmerged baseline.
The development rows are training-seen; labels never select a kernel. This is
neither a held-out quality evaluation nor a production arrival-rate/SLO result.

Use the existing NC2 B200, without allocating capacity. Reuse the completed
merged BF16 results and stop its idle server; the user explicitly questions the
need to reserve another 49 GB of GPU memory for an already measured control.
The FP8 candidate uses localhost 8010. Keep the merged checkpoint on ephemeral
disk; online quantization adds no persistent weight artifact. Reuse shared disk
compiler/kernel caches and record new cache keys/runtime/worker PIDs.

At model loading, audit all 64 fused MLP projection dtypes/methods/resolved GEMM
classes and require BF16 in every other loaded linear. Reject weight-only FP8
fallback, incorrect precision scope or incomplete layer coverage. Bind all
executed code/config/input/merged-artifact checksums. Run a fresh twenty-row
canary against the archived master, unmerged and merged BF16 references with
unchanged mean score difference <=0.02, correlation >=0.99 and nonzero adapter
effect. Stop before timings if it fails; preserve the failure and ask only if a
diagnostic continuation needs separate authorization. Stop on OOM, compiler or
backend failure, nonfinite/missing score, input drift or truncation.

Report prompt tokens/s first, latency and requests/s, paired repeat-median
score/margin differences and threshold flips against the merged control, and
within-candidate variation. A >10% warmed gain merits follow-up; passing a small
canary is not enough to promote production precision. Do not repeat the BF16
control, rerun full ID, tune concurrency or launch a broad kernel sweep here.
Retain a successful candidate for compatible work. Do not keep a separate
resident control unless a new matched control measurement needs it.

Historical SM120 experiments motivate this first choice: MLP-only native FP8
passed a different adapter/workload's checks and improved full-split throughput
1.26x, whereas tested FP4 inference layouts failed score gates. Those results
do not establish speed or fidelity on this B200/final FP4-trained adapter.
See `docs/findings/blackwell_inference_search.md`.

```bash
python -m experiments.b200_inference_kernels.run --output fp8_mlp02
```

Results: `results/b200_inference_kernels/`; logs:
`logs/runpod/b200_inference_kernels/`. Inspect startup every 30–60 seconds.
No in-chat scheduling tool is available; active-turn checks cannot promise an
agent wakeup after the turn ends.

The first attempt, `fp8_mlp01`, stops before model loading because port 8001
already has a listener. Preserve its receipt; the retry uses verified-free port
8010 and leaves the existing listener untouched. No GPU timing or numerical
measurement comes from the failed launch.

## Completed native FP8 MLP screen

`fp8_mlp02` completes all six matched passes. The loading audit verifies 64
fused MLP projections using `Fp8PtpcOnlineLinearMethod` and
`CutlassFP8ScaledMMLinearKernel`, with all other loaded linears BF16. Model
loading uses 5.96 GiB versus 7.99 GiB for BF16. The serving engine still reserves
49,798 MiB overall under the unchanged memory fraction, including cache and
runtime allocations; lower weight memory is not equal to total server savings.

Each statistic below is the median of two repeats on the same 64 prompts,
269,411 input tokens, with prefix caching off. Conversion cost is included.

| Concurrency | Merged BF16 input tokens/s | FP8 MLP input tokens/s | Throughput gain | FP8 p50 / p95 latency (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 29,763 | 32,105 | 7.87% | 0.0973 / 0.3382 |
| 4 | 69,049 | 78,356 | 13.48% | 0.1984 / 0.3821 |
| 16 | 107,073 | 115,423 | 7.80% | 0.5244 / 0.8681 |

Median requests/s are 7.627 / 18.614 / 27.419 and pass durations
8.406 / 3.438 / 2.334 seconds. Interactive p50 falls 0.1041 → 0.0973 seconds.
Only concurrency 4 clears the predeclared >10% interest threshold. This is a
modest end-to-end gain with conversion cost included, not a kernel-only speedup.
Two short repeats are screening evidence; do not infer universal acceleration
or compute/memory limitation without profiling.

Fresh twenty-row canary passes: mean score difference/correlation are
0.00493345/0.99966835 versus archived master, 0.00398037/0.99983862 versus
dynamic LoRA and 0.00339731/0.99980108 versus merged BF16. Nonzero adapter effect
against explicitly reused base scores is 0.79854948. On 64 paired repeat medians,
mean absolute score drift at concurrency 1/4/16 is
0.010563/0.010378/0.010577, maximum 0.103560 throughout; mean margin drift is
0.091797/0.092773/0.091797, maximum 0.5. One 0.5-threshold decision changes in
each comparison, the same identity. Across all candidate arrays the mean/max
per-row range is 0.000867/0.027824 with zero threshold-unstable rows. These
differences do not establish held-out quality parity or promote FP8 production
precision. Keep merged BF16 as the comparison reference for further screens.

Readiness is 698.666 seconds, fresh canary 10.622 and four-length warmup 0.452,
excluded from all timed passes. Native graph compilation takes 114.03 seconds
and initial runtime warmup 93.99 seconds; checkpoint loading takes 1.67 seconds.
Fifteen focused tests and Ruff pass; local collection verifies every result
checksum, row identity, prompt hash, token count, finite score and native audit.
Preserve the first port-collision failure and the BF16 retirement receipt.

Only candidate API PID 73262 / engine PID 73329 remains resident at localhost
port 8010, healthy and idle. The old BF16 server is stopped, original merged
checkpoint/caches/results preserved and pod kept running. Reuse compatible
candidate request trials with a new output directory and `--reuse-server`.
