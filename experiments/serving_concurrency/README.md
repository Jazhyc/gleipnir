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

**Layout `fast_g2048e`** (each multiple of 32 from 288 to 768 and the
size one below it: 287, 288, 319, 320, … 767, 768; the
default sizes to 256 and the 27B sizes from 896) keeps each row count's
remainder class. Against the release at c1 it is bit-identical on all 3,012
ID rows, 6,395 OOD rows and 320 benchmark rows, with c1 136,574 tok/s, p50
11.6 ms, p95 120 ms and c128 227,771 tok/s (release 102,582, 29.0, 119 and
226,391). Its concurrency curve (`breakdown01/sweep_e`; tok/s, p50 ms): c1
137,790, 11.5; c2 163,186, 34.6; c4 191,308, 69.9; c8 209,482, 160; c16
225,109, 283; c32 229,031, 568; c64 228,205, 1,119; c128 227,709, 2,158.
Scores at higher concurrency were not compared; they already vary with
batch composition. The user adopted the layout for the report on
2026-11-15; the release image and command are to follow. The switch at
multiples of 32 was observed only on this software stack and its kernel is
not identified, so every image rebuild must repeat the bit-identity check
(`fast_g2048e` against `fast` at concurrency 1 on short prompts). 399 result and log files of the session were pulled and
checksum-verified.

## Prefix caching and score identity

User request, 2026-11-15, same Pod, 19:45–20:06 UTC: do cached scores equal
uncached scores, and if not, why. `prefix.py` scores each `full320` prompt
at concurrency 1 after a cache reset (`uncached`), again (`replay`), and
after a reset and a request with its first half (`half`), on the 4B with
`--enable-prefix-caching --mamba-cache-mode align`
(`results/serving_concurrency/prefix01/`; the released-stack run is in
`rerun/`, `trial/` holds nothing, and the first attempt failed on a log
name).

| Step | Replay, 149 hit rows | Half, 126 hit rows | Half, 194 rows without hit |
|---|---|---|---|
| `bf16` | 0 changed | 32 changed, mean 0.004, max 0.054 | 0 changed |
| `quant` | 0 changed | 100 changed, mean 0.025, max 0.210 | 1 changed (0.011) |
| `fast` | 0 changed | 103 changed, mean 0.024, max 0.224 | 1 changed (0.005) |

- **The cache is lossless**: an identical replay is bit-identical on every
  step, with up to 99% of its tokens cached.
- **The split of the prefill into steps is what changes scores.** vLLM sets
  the attention block to 528 tokens for the 4B so that an attention page is
  as large as a GDN state page, and align mode saves GDN state only at
  block boundaries. A resumed request computes the rest in other steps than
  an uncached one.
  - Without a cache, `--max-num-batched-tokens 4096` changes `fast` on 67 of
    87 prompts above 4,096 tokens (mean 0.017, maximum 0.342) and on none
    below. (`bf16` with that budget also changes 62 of 233 shorter prompts,
    mean 0.004; not explained.)
  - Enabling caching alone changes prompts above 528 tokens against the
    server without it (`fast` 115 of 149, mean 0.021; `quant` 118; `bf16`
    21, mean 0.002) and no shorter prompt.
  - With `--max-num-batched-tokens 528` every request advances in the same
    steps, and `half` equals `uncached` on all 127 hit rows for `fast` and
    `bf16`. One uncached pass over the 320 prompts then takes 72.2 s
    against 19.3 s (`fast`).
- BF16 is not split-invariant either (rounding); the low-precision path
  amplifies it about sixfold, from its first rung. The kernel responsible
  was not identified. ID/OOD quality under caching was not measured: those
  prompts are independent and would not hit the cache.

**ID/OOD through the cache** (user request; `prefix.py --cache --splits ood
id`, `prefix01/prefix_4b_fast_cache_eval`, 20:10–20:32 UTC). Released 4B,
caching on, normal step size, concurrency 1. Each row starts from an empty
cache, scores the first half of the prompt text (the midpoint decision; not
a well-formed monitor prompt, it only fills the cache) and then the whole
prompt. pAUROC@20 of the final scores against the same rows uncached at
concurrency 1 without caching (`breakdown01/parity_4b_fast_c1`):

| Source | Uncached | Cached | Rows with a hit |
|---|---:|---:|---:|
| ID macro | 0.9098 | 0.9113 | 3,007 of 3,012 |
| Gloom exfiltration | 0.8473 | 0.8503 | |
| STRIDE | 0.9724 | 0.9723 | |
| OOD macro | 0.7669 | 0.7661 | 5,068 of 6,395 |
| AgentDojo | 0.8152 | 0.8037 | 190 of 1,500 |
| CUA-SHADE-Arena | 0.7599 | 0.7530 | 950 of 955 |
| IaC | 0.8235 | 0.8352 | 1,433 of 1,443 |
| MLE-Sabotage | 1.0000 | 1.0000 | 148 of 148 |
| Rogue-Eval | 0.9547 | 0.9562 | 1,499 of 1,499 |
| SHADE-Arena | 0.2479 | 0.2484 | 848 of 850 |

About half of all prompt tokens came from the cache. Among rows with a hit,
82% of scores change (ID mean 0.025, maximum 0.40, 71 cross 0.5; OOD mean
0.027, maximum 0.51, 115 cross 0.5). Of the OOD rows without a hit, 1,016 of
1,327 also change, because caching splits prompts above 528 tokens; this
is most of AgentDojo's change. The macro scores move by +0.0015 and
−0.0008, and no source by more than 0.012: individual scores change, the
ranking quality does not.

## Tokenizer

User request, 2026-11-15, same Pod, 20:37–21:44 UTC: the end-to-end effect
of Gigatoken against the tokenizer of Transformers.

- **Gigatoken was not in use on vLLM 0.31.** `gleipnir.serving.gigatoken`
  replaces `BaseRenderer._encode`; vLLM 0.31 has no such method and
  tokenizes in `BaseRenderer._tokenize_prompt`. The first attempt
  (`tokenizer01`, the SIGUSR1 switch of `GLEIPNIR_GIGATOKEN_AB`) counted
  zero calls of the patched encoder in either mode over about 6,000
  requests. Every vLLM 0.31 measurement and the released image therefore
  tokenize with Transformers, although the startup log reports the native
  frontend.
- **Hook.** `tokenizer_hook/sitecustomize.py` routes
  `_tokenize_prompt` through the installed native encoder while a mode file
  says `native` (`GLEIPNIR_TOKENIZER_HOOK`, `--gigatoken` in `run.py` and
  `lengths.py`). It is experiment code; the user decided to add Gigatoken
  as the last serving step, so the deployment stack needs an equivalent.
- **Direct timing** on the Pod CPU, `full320`: Transformers 1.262 s,
  Gigatoken 0.020 s; median 0.49 against 0.01 ms for 257–1,024 tokens and
  15.3 against 0.2 ms above 8,192 tokens; identical IDs on all 320 prompts.
- **Matched comparison** (`tokenizer.py`, `tokenizer02`): one server,
  alternating passes, three rounds (27B `bf16` two). Scores at c1 are
  identical in every condition. Median tok/s at c128; tok/s, p50 and p95 ms
  at c1:

  | Condition | Transformers | Gigatoken |
  |---|---|---|
  | 4B `fast_g2048e` | 225,455; 133,051, 12.9, 123 | 225,560; 160,931, 10.7, 98 |
  | 4B `bf16` | 119,126; 83,571, 18.9, 207 | 119,179; 92,104, 17.9, 182 |
  | 27B `fast` | 52,984; 44,286, 22.6, 444 | 53,135; 46,233, 21.3, 422 |
  | 27B `bf16` | 22,933; 20,954, 35.1, 991 | 22,877; 21,364, 34.9, 969 |

  The server counted 9.1 to 9.8 s inside Transformers over 1,984 prompts
  (4.6 to 4.9 ms each) and about 0.2 s inside Gigatoken over 2,304. The
  Transformers passes of `fast_g2048e` here are slower than `sweep_e`
  without the hook (12.9 against 11.5 ms); whether the hook or the run
  causes this is not established.
- **Sweeps with Gigatoken** (`giga01`, for the report's figure and last
  table row): `4b_fast_g2048e_giga` c128 227,509 tok/s, p50 2,230 ms; c1
  161,699, 10.4, 98.2; c16 225,396, 281; peak 229,608 at c32.
  `27b_fast_giga` c128 52,957, 9,362; c1 46,728, 21.3, 421; c16 52,890,
  1,197; peak 53,119 at c32. Single requests: 4B 2.94 s at 262,144 and
  36.3 s at 1,048,576 tokens; 27B 10.75 s and 125.9 s. The 4B length sweep
  ran with the `e` graph sizes at the 262,144 context without problems.

All tokenizer results and logs were pulled and checksum-verified
(`tokenizer02/manifest.sha256`, `giga01/manifest.sha256`).

## Agent-trajectory benchmark (the report's benchmark)

User decisions, 2026-11-15/16, same Pod, 22:06–00:21 UTC:

- **Benchmark.** `full320` is a sample of the training mix: 224 of its
  prompts are deception data or CoT Red-Handed (median 387 tokens), so its
  median latency describes short single prompts. The report now uses the 96
  multi-step agent trajectories (`run.py --agent`: BashArena 42, BashBench
  22, STRIDE 20, Gloom 12; 1,193,325 tokens, mean 12,430, median 10,067,
  quartiles 5,595 and 18,187, range 1,705–28,733), in the frozen order.
  `agent01` (130 prompts, with CoT Red-Handed) and `agent02` (started while
  an old server held the GPU; the launcher refused) hold nothing usable.
- **Concurrency.** With 96 prompts the sweep stops at 64. Throughput
  saturates at 8, which the report's table uses (user); intermediate steps
  were timed at 64, 16 and 1 first and at 8 in `agent03/c8`.
- **4B graphs up to 4,096** (`fast_g4096e`: the `e` sizes plus 2,560, 3,072,
  3,584, 4,096; user, to remove the latency step at 4,096 tokens). Checked
  first: ID and OOD at concurrency 1 are bit-identical to `fast`
  (`parity4096`; 0 of 9,407 scores change, 1,443 rows are in the new
  range). A 4,096-token prompt takes 22.5 ms against 26.3 ms.
- **27B graphs up to 4,096** (`27b_fast_g4096_giga`): no gain (4,096
  tokens 77.2 against 77.7 ms; benchmark c1 191.4 against 192.0 ms), so
  the 27B stays at 2,048.

Medians of three passes (`agent03/summary.json`, `agent03/c8/summary.json`);
tok/s and p50 ms at c8, then tok/s, p50 and p95 ms at c1:

| Condition | c8 | c1 |
|---|---|---|
| 4B `bf16` | 120,284; 816 | 103,266; 94.4; 268 |
| 4B `quant` | 216,499; 450 | 160,069; 61.2; 168 |
| 4B `quant_g4096e` | 216,286; 451 | 156,455; 63.9; 170 |
| 4B `lastquery_g4096e` | 220,061; 443 | 158,629; 62.5; 168 |
| 4B `route_g4096e` | 218,814; 448 | 158,908; 63.5; 167 |
| 4B `fast_g4096e` | 231,065; 421 | 174,029; 57.8; 156 |
| 4B `fast_g4096e_giga` | 230,375; 424 | 204,673; 48.0; 133 |
| 27B `bf16` | 22,950; 4,271 | 21,720; 452; 1,288 |
| 27B `quant_nograph` | 51,177; 1,910 | 46,717; 208; 606 |
| 27B `quant` | 51,125; 1,913 | 46,354; 210; 607 |
| 27B `lastquery` | 51,365; 1,908 | 46,641; 207; 614 |
| 27B `fast` | 53,408; 1,831 | 48,535; 199; 581 |
| 27B `fast_giga` | 53,356; 1,834 | 50,788; 192; 560 |

- Graphs and the GDN route do not show on this benchmark (one prompt is
  short enough for a graph). The 4B rungs with graphs to 2,048
  (`*_g2048e`) agree with those to 4,096 within 2%.
- Sweeps (tok/s, p50 ms) for c1, 2, 4, 8, 16, 32, 64: 4B
  `fast_g4096e_giga` 204,673/48; 212,418/113; 224,844/207; 230,375/424;
  230,205/846; 228,400/1,703; 230,317/2,671. 4B `bf16` 103,266/94;
  114,597/210; 119,175/391; 120,284/816; 120,254/1,628; 119,992/3,258;
  120,327/4,999. 27B `fast_giga` 50,788/192; 51,236/467; 52,769/887;
  53,356/1,834; 53,373/3,681; 53,359/7,334; 53,298/11,234. 27B `bf16`
  21,720/452; 22,296/1,064; 22,800/2,056; 22,950/4,271; 22,967/8,544;
  22,964/17,067; 22,979/26,828.
- 4B length sweep with graphs to 4,096 (`giga02`): 10.7, 10.6, 10.8, 14.5,
  22.5, 38.5, 73.2, 157.9, 375.5, 989.4 ms from 256 to 131,072 tokens;
  2.93 s at 262,144; 9.76 s and 36.3 s with YaRN.
- Costs for the report's Table 1 use c8: 4B 120,284, 216,499, 231,065;
  27B 22,950, 51,177, 53,408 tok/s.

Results and logs were pulled and checksum-verified
(`agent03/manifest.sha256`).
