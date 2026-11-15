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

## Per-change breakdown and 4B graph sweep

User request, 2026-11-15: the gain of each change for the 27B, and whether
the 4B benefits from larger CUDA graphs (the 4B release captures vLLM's
default sizes up to 256 tokens, the 27B up to 2,048), with a sweep of graph
sizes and a 4B breakdown at the best one. Written before the run.

- **27B ladder** (cumulative, one server each): `bf16`; `quant_nograph`
  (`QuantNoTrimWorker` without `--compilation-config`); `quant` (+ graphs to
  2,048); `lastquery` (`QuantLastQueryWorker`: + last-query attention);
  `fast` (+ lean GDN host path, the release command).
- **4B graph sweep.** `fast` (matched control) and `fast_g<cap>` for caps
  512, 1,024, 2,048, 4,096 and 8,192: the release command plus piecewise
  capture sizes up to the cap (the 27B sizes to 2,048, then four per
  power-of-two interval, padding at most 25%; sizes 1,535 and 8,191 keep
  padding from crossing the FP4 producer and GDN kernel switches).
- **Selection** (frozen): the cap with the lowest median c1 latency on
  `full320` among caps whose c128 throughput is at least 99% of the matched
  `fast`; caps within 2% of the best latency tie and the smallest wins. If no
  cap lowers the c1 median by more than 5%, the answer is that graphs do not
  help and no breakdown follows.
- **4B breakdown** at the selected cap: `quant_g<cap>`, `lastquery_g<cap>`,
  `route_g<cap>`, with `bf16` and `fast_g<cap>` as its ends.
- **Workload and measures** as above, at concurrency 128 and 1 only
  (`--ends`), in `results/serving_concurrency/breakdown01/`
  (`--session breakdown01`). Also c1 latency by prompt-length band, and the
  shift of c1 scores of each graph condition against `fast` (mean and
  maximum absolute difference, rows crossing 0.5). Larger graphs change the
  serving recipe, so a selected cap is then scored on the frozen ID and OOD
  workloads of `experiments/release_serving/validate.py` and compared with
  the recorded release rows (user, 2026-11-15). No cap is adopted for the
  release here; that stays the user's decision.
- **Stop** as above; a graph condition that fails to start or audit is
  recorded and skipped.
- **Compute.** One EU-RO-1 B200 Pod from the same image, authorized by the
  user on 2026-11-15.

### Result

Run 2026-11-15, 16:55–18:39 UTC, Pod `179xu9pmh64fxh` (EU-RO-1, the same
machine and B200 `GPU-e5eb9962…` as the first session). All conditions
completed on the first start. Added during the run at the user's request:
stock BF16 with graphs to 2,048 for both models (`bf16_g2048`), the full
concurrency sweep of `fast_g2048` (`breakdown01/sweep2`; `sweep_ends_only`
is an earlier attempt that timed only 128 and 1), and ID/OOD scoring at
concurrency 1 (`parity.py --concurrency 1`). Results and logs were pulled
and checksum-verified (`breakdown01/manifest.sha256`). Medians of
three passes (`breakdown01/summary.json`):

| Condition | c128 tok/s | vs previous | c1 tok/s | c1 p50 / p95 (ms) |
|---|---:|---:|---:|---:|
| 27B `bf16` | 22,924 | | 21,007 | 34.4 / 988 |
| 27B `bf16_g2048` | 22,975 | | 20,883 | 36.7 / 987 |
| 27B `quant_nograph` | 50,849 | 2.218× | 35,537 | 58.6 / 458 |
| 27B `quant` | 50,682 | 0.997× | 42,275 | 24.0 / 461 |
| 27B `lastquery` | 50,981 | 1.006× | 42,093 | 24.7 / 462 |
| 27B `fast` | 52,922 | 1.038× | 44,660 | 22.5 / 443 |
| 4B `bf16` | 119,728 | | 84,742 | 19.0 / 205 |
| 4B `bf16_g2048` | 119,858 | | 89,175 | 15.0 / 202 |
| 4B `quant_g2048` | 212,456 | 1.774× | 120,927 | 15.3 / 128 |
| 4B `lastquery_g2048` | 215,660 | 1.015× | 122,209 | 15.3 / 125 |
| 4B `route_g2048` | 214,885 | 0.996× | 126,641 | 13.4 / 130 |
| 4B `fast_g2048` | 227,457 | 1.059× | 138,839 | 11.3 / 118 |

4B graph sweep on the release command (`fast`, graphs to 256: 226,391 tok/s
at c128; c1 102,582 tok/s, p50 29.0 ms):

| Cap | c128 tok/s | c1 tok/s | c1 p50 (ms) |
|---|---:|---:|---:|
| 512 | 214,337 | 125,057 | 12.8 |
| 1,024 | 226,498 | 128,430 | 12.2 |
| 2,048 | 227,457 | 138,839 | 11.3 |
| 4,096 | 222,838 | 138,022 | 11.9 |
| 8,192 | 226,814 | 139,761 | 11.2 |

- **Selection: 2,048** (8,192 is within 2%; 512 and 4,096 miss the c128
  bound because one of their three passes ran at about 143K tok/s, as did
  one pass of `fast` and of `fast_g2048`; cause not identified).
- **Latency by length at c1.** Graphs remove the fixed cost for every length
  they cover: 256–1,024 tokens 28–29 → 11 ms, 1,024–2,048 tokens 30 → 14 ms;
  nothing changes above 4,096 tokens. The 27B shows the same effect
  (`quant_nograph` 58.6 ms against 24.0 ms). BF16 gains for the 4B (19.0 →
  15.0 ms) and not for the 27B (34.4 → 36.7 ms). Throughput at c128 does not
  change with graphs.
- **Per change.** 27B: last-query 1.006×, GDN host path 1.038×. 4B gains
  with graphs equal the gains without them.
- **`fast_g2048` across concurrency** (tok/s, p50 ms): c1 137,705, 11.5; c2
  163,407, 34.5; c4 191,181, 70.9; c8 212,446, 149; c16 223,652, 279; c32
  227,589, 575; c64 226,622, 1,132; c128 227,018, 2,151.
- **Scores.** c1 scores are deterministic. With graphs to 2,048 they change
  only for prompts of 257–767 tokens that are padded to a captured size;
  exact-size prompts and all prompts of 769–2,048 tokens are bit-identical.
  On `full320`: 135 rows change, mean 0.027, maximum 0.458, 5 cross 0.5 (the
  release stack's own c1-to-c128 difference is mean 0.039, maximum 0.44).
- **ID/OOD pAUROC@20, 4B** (recorded release 0.9117 / 0.7703):

  | Scoring | `fast` ID / OOD | `fast_g2048` ID / OOD |
  |---|---:|---:|
  | batches of 128 (`validate.py`) | 0.9110 / 0.7708 | 0.9104 / 0.7712 |
  | concurrency 1 | 0.9098 / 0.7669 | 0.9098 / 0.7639 |

  Batches of 128 rarely replay a graph. At concurrency 1 the ID rows are
  identical (14 of 3,012 are short enough for a graph). On OOD 499 of 6,395
  scores change, 496 of them in AgentDojo, whose pAUROC@20 falls from 0.8152
  to 0.7973; the other five sources are unchanged.

### Diagnosis of the score change (same Pod, 18:43–19:40 UTC)

The change does not come from padding. Below 768 rows a step's scores take
one of two values, by whether its row count is a multiple of 32:

- Layouts `fast_g2048` (sizes 320, 384, … 768) and `fast_g2048o` (288, 352,
  … 704) give identical scores on all 1,500 AgentDojo rows.
- `fast_g2048d` (every 16 from 272) equals `fast_g2048` on all 320 rows of
  257–767 tokens whose padded size is a multiple of 32, and equals the
  release on 278 of the 279 others.
- Without larger graphs, the 24 rows whose own length is a multiple of 32
  already score as with graphs.
- Stock BF16 has the same switch with a small effect (`bf16` against
  `bf16_g2048`: 37 of 152 benchmark rows of 257–768 tokens, mean 0.003;
  OOD at c1 0.8042 against 0.8044, AgentDojo 0.9466 against 0.9478), and the
  27B too. The kernel responsible was not identified. The quantized 4B
  amplifies it from its first low-precision rung on.

AgentDojo is where the quantized 4B is sensitive to rounding: 0.8425 in
batches of 128, 0.8152 at c1, 0.8070 with `fast_g2048d` (half the short
rows switched), 0.7973 with `fast_g2048`, against 0.9466 in BF16. The
release's reported OOD scores are for batched scoring.

**Layout `fast_g2048e`** (sizes 32k − 1 and 32k from 287 to 768, the
default sizes to 256 and the 27B sizes from 896) keeps each row count's
remainder class. Against the release at c1 it is bit-identical on all 3,012
ID rows, 6,395 OOD rows and 320 benchmark rows, with c1 136,574 tok/s, p50
11.6 ms, p95 120 ms and c128 227,771 tok/s (release 102,582, 29.0, 119 and
226,391). Its concurrency curve (`breakdown01/sweep_e`; tok/s, p50 ms): c1
137,790, 11.5; c2 163,186, 34.6; c4 191,308, 69.9; c8 209,482, 160; c16
225,109, 283; c32 229,031, 568; c64 228,205, 1,119; c128 227,709, 2,158.
Scores at higher concurrency were not compared; they already vary with
batch composition. Adopting the layout for the 4B release is the user's
decision. 399 result and log files of the session were pulled and
checksum-verified.
