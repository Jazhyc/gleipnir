# Serving throughput and latency by step and concurrency

User request, 2026-11-15: the v1 report needs prompt throughput and request
latency after each serving step for both released monitors, and how both
change with client concurrency, measured on one host. Existing records give
the per-step throughput at concurrency 128 only
([4B ladder](../gleipnir4b_serving_ladder/README.md),
[27B](../qwen38_27b_serving/README.md)); the only concurrency curve is an
older 4B recipe ([score scaling](../../docs/findings/b200_score_scaling.md)).

- **Question.** For Gleipnir 4B v0.4 and 27B v0.2 on one B200, what are
  prompt tokens/s and request latency (p50, p95) at client concurrency 1 to
  128 with stock BF16 serving and with the released stack, and what does each
  serving step add at concurrency 128 and 1?
- **Conditions** (one fresh server each; released merged checkpoints):
  - 4B: the five cumulative rungs of the 4B ladder (`bf16`, `quant`,
    `lastquery`, `route`, `fast`); `fast` must equal the release command.
  - 27B: `bf16`; `quant` (the release command with `QuantNoTrimWorker`:
    FP8/FP4 projections, MXFP8 attention, tuned tiles and fused producers,
    without last-query attention and the lean GDN host path); `fast` (the
    release command at its recorded 32K context).
- **Workload.** Frozen `full320` (1,310,581 prompt tokens with the 4B
  compact instruction) through `/v1/monitor/score`, prefix caching off,
  closed loop. After a `quick64` and a `full320` warmup at concurrency 128:
  - `bf16` and `fast`: three timed passes at each of concurrency 1, 2, 4, 8,
    16, 32, 64 and 128, from 128 down;
  - other steps: three timed passes at concurrency 128 and at 1.
  Every condition also runs `quick64` at concurrency 1 three times, for
  continuity with earlier records.
- **Prompt length** (`lengths.py`; user request: up to the maximum context
  of about one million tokens). `bf16` and `fast` of each model time single
  requests on synthetic plain-text prompts of exact token length, in two
  further servers each, as in `b200_context_million` and the 27B
  `long_context.py`:
  - `native`: checkpoint RoPE, `--max-model-len 262144`, lengths 256 to
    262,144 (powers of two);
  - `yarn`: Qwen's YaRN with factor 4, `--max-model-len 1048576`, lengths
    262,144 (overlap), 524,288 and 1,048,576. YaRN changes scores, so scores
    there are timing payloads and not validated monitor outputs.
  One excluded warmup per length, then five requests (three from 131,072).
  The 32,768-token chunk budget is unchanged. A length that fails is
  recorded and ends its phase.
- **Measures.** Median over passes of prompt tokens/s and of per-pass p50 and
  p95 request latency. Latency starts when the client slot is acquired and
  includes localhost HTTP, tokenization, queueing and inference.
- **Environment.** `environment.py` records datacenter, GPU, driver, CPU,
  CPU and memory limits, kernel and package versions, for the report.
- **No selection.** Descriptive only; the workload is training-seen, so
  scores are saved for a sanity check and not reported as quality.
- **Stop** on a failed startup (the known MXFP8 compile race is retried
  twice), a `fast` command that differs from the release command, a failed
  BF16 audit, a non-finite score or completion. If the 27B `quant` worker
  fails its audits, skip that condition and record the failure.
- **Compute.** One EU-RO-1 B200 Pod from `ghcr.io/jazhyc/gleipnir-serve:0.2`,
  authorized by the user on 2026-11-15.

```bash
PY="/tmp/gleipnir-vllm031-runtime/bin/python -m experiments.b200_vllm031.runtime"
$PY -m experiments.serving_concurrency.environment
$PY -m experiments.serving_concurrency.run --model 4b --step bf16
$PY -m experiments.serving_concurrency.lengths --model 4b --step bf16 --phase native
python -m experiments.serving_concurrency.summarize
python -m experiments.serving_concurrency.plot
```

## Result

Run 2026-11-15, 02:28–04:35 UTC, Pod `nsw57y1tcxh9gw` (EU-RO-1, B200
`GPU-e5eb9962…`, driver 595.91.07, EPYC 9555, 28 vCPU and 283 GB allotted;
`results/serving_concurrency/environment.json`). All 16 conditions completed
without retries or errors, including the 27B `quant` worker. Results and logs
were pulled and checksum-verified (240 files, `manifest.sha256`), and the Pod
was deleted (HTTP 204, lookup 404). Medians of three passes
(`results/serving_concurrency/summary.json`):

| Condition | c128 tok/s | vs previous | c128 p50 (s) | c1 tok/s | c1 p50 / p95 (ms) |
|---|---:|---:|---:|---:|---:|
| 4B `bf16` | 119,329 | | 4.11 | 84,885 | 18.8 / 205 |
| 4B `quant` | 209,757 | 1.758× | 2.33 | 88,052 | 34.2 / 136 |
| 4B `lastquery` | 214,529 | 1.023× | 2.27 | 90,836 | 33.3 / 129 |
| 4B `route` | 213,899 | 0.997× | 2.29 | 95,420 | 31.5 / 127 |
| 4B `fast` | 226,278 | 1.058× | 2.15 | 99,125 | 29.2 / 122 |
| 27B `bf16` | 22,949 | | 21.42 | 20,993 | 34.5 / 989 |
| 27B `quant` | 50,675 | 2.208× | 10.00 | 42,009 | 24.6 / 461 |
| 27B `fast` | 52,864 | 1.043× | 9.55 | 44,112 | 23.0 / 443 |

- Throughput saturates near concurrency 16. Peaks: 4B `fast` 227,765
  (c64), 4B `bf16` 119,700 (c32), 27B `fast` 53,071 (c32), 27B `bf16`
  22,957 (c32).
- The 4B gains agree with the US-CA-2 ladder (1.77×, then 1.07× in total;
  here 1.76× and 1.08×). The 27B `quant` step is measured here for the first
  time: 2.21× and then 1.04×, against 2.25× and 1.02× derived for the report's
  cost table; the total (2.30×) is the same. The 27B `quant` command keeps
  the release command's CUDA graphs.
- At concurrency 1 the released 4B has a higher median latency than BF16
  (29.2 against 18.8 ms) and a lower p95. Single synthetic requests show a
  fixed cost: the released 4B takes about 29 ms from 512 to 4,096 tokens,
  BF16 18 to 22 ms up to 2,048 tokens; they cross near 3,000 tokens. The
  released 27B is faster than BF16 at every length.
- Single-request latency (median): at 262,144 tokens 3.25 s (4B `fast`),
  6.52 s (4B `bf16`), 11.06 s (27B `fast`), 24.82 s (27B `bf16`); at
  1,048,576 tokens with YaRN 37.9 s, 90.0 s, 127.8 s and 297.2 s. The YaRN
  server reproduces the native latency at 262,144 tokens within 1.5%.
