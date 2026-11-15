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
```
