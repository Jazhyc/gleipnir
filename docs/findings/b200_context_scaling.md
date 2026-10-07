# B200 fixed 2K context scaling

2026-10-07. The unchanged selected reference reaches a median 252,647 input
tokens/s and 123.36 requests/s with exactly 2,048-token prompts. This is 16.56%
more input throughput and 2.33 times the request throughput of the earlier
mixed-length full320 peak. Concurrency 16 already reaches 97.20% of the new
measured peak; concurrency 32 gives the highest and most stable measured median.
Higher concurrency mostly increases waiting. No serving or precision changes.

## Contract and controls

The user chose prompts concentrated near 2,048 tokens and added concurrency 8.
Full320 has no natural prompts in the 1,792–2,304 interval. Derive 320 distinct
text windows from 95 longer original prompts, decode/re-encode with the frozen
checkpoint tokenizer, and freeze exact source/window lineage before timing.
Every prompt is exactly 2,048 tokens; each pass processes 655,360 input tokens.
Labels are removed because cropping does not preserve their meaning. This is an
artificial systems workload, not monitor quality or observed production traffic.
See [the experiment contract](../../experiments/b200_context_scaling/README.md).

Use the same ordered windows at all eight levels, one excluded warmup and three
timed repeats per level with fresh HTTP pools. Closed-loop request latency
excludes semaphore waiting and includes localhost HTTP, text encoding, engine
waiting and inference. Native endpoint validation checks exact token counts,
finite logits/margins/scores and their consistency on every request.

Bind the same live process/command/source identities to the completed
[reference sweep](b200_score_scaling.md). Reuse the warm EU-RO-1 B200 scorer with
native Gigatoken/direct FROST, FP4 projections/MLPs, MXFP8 attention prefill,
BF16 operands/KV and FP32 gates/state, causal LAST pooling, upstream FCFS,
128-sequence cap, 32768-token budget and prefix caching off. No restart,
capacity operation or baseline promotion. The unchanged real adapter canary
passes: mean absolute score difference 0.000807, correlation 0.999967 and
nonzero adapter effect 0.839906. Cropped-window AUROC is deliberately undefined.

## Measurements

Medians across three timed repeats:

| Client concurrency | Input tokens/s | Requests/s | p50 latency | p95 latency | p99 latency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 66,822 | 32.63 | 30.36 ms | 32.77 ms | 33.78 ms |
| 2 | 75,616 | 36.92 | 53.80 ms | 56.53 ms | 59.38 ms |
| 4 | 147,731 | 72.13 | 55.09 ms | 56.75 ms | 58.34 ms |
| 8 | 192,075 | 93.79 | 84.15 ms | 87.95 ms | 92.62 ms |
| 16 | 245,576 | 119.91 | 130.24 ms | 155.90 ms | 162.43 ms |
| 32 | 252,647 | 123.36 | 253.32 ms | 284.66 ms | 294.17 ms |
| 64 | 250,821 | 122.47 | 507.80 ms | 532.20 ms | 640.95 ms |
| 128 | 247,669 | 120.93 | 1012.23 ms | 1063.65 ms | 1099.39 ms |

Input-throughput repeat ranges in level order: 63,991–67,531;
72,946–75,918; 147,411–147,807; 188,380–193,333; 229,137–246,210;
251,945–252,762; 232,841–251,108; 231,346–248,944 tokens/s. Preserve the
slower repeats at c16/64/128; three repeats do not establish a precise optimum.
From c16 to c32, median input throughput increases 2.88% while p95 latency
increases 82.59%. At c1, p50 is close to the earlier 30.47 ms, while p95 falls
from 130.74 to 32.77 ms because the long-input tail is absent.

Also measure the missing original full320 c8 point on this same warm reference,
with one excluded warmup and three timed repeats. Median input throughput is
196,514 tokens/s (195,353–197,887), request throughput 47.98/s, p50 153.28 ms,
p95 303.93 ms and p99 361.99 ms. Store the addendum separately; the original
reference report remains immutable. Both plotted curves now include c8.

The prior workload averages 4,095.57 tokens but has median 503.5 and maximum
28,733: 31 prompts at least 16K long contribute 53.02% of its tokens. The
comparison changes both average length and distribution. It cannot isolate a
length-only causal effect or establish parity with another model's benchmark
without matching its serving, hardware, workload and latency definition.

## Artifacts and closure

`results/b200_context_scaling/twok01/` preserves 24 timed 2K prediction passes,
three original full320 c8 passes, frozen windows/lineage, source snapshots,
canary/native receipts, reference binding and logs. All 40 collected remote
artifacts match their checksums. `scaling.json`/`scaling.csv` contain medians and
ranges; `scaling.png`/`scaling.svg` compare throughput, requests/s and p95 with
repeat-range shading. Analysis source/raw-report checksums are bound in JSON.
Four focused window/identity/export tests pass; Ruff is clean.

Closure health is 200. API PID 27565 and sole GPU engine 27588 remain unchanged,
with 172288 MiB GPU allocation and 0% utilization after timing. Leave the
reference warm and persistent compiler/kernel caches intact.
