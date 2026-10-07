# Tokenization contribution to B200 serving

2026-10-07. **The text frontend materially affects long-prompt interactive
latency, but bypassing it produces little saturated-throughput improvement.**
Keep the selected Direct FP4/MXFP8 kernel reference unchanged. This is a client
ablation on the resident server, not a new precision or production recipe.

## Matched serving results

Use exact token IDs from the same rendered prompts, with no added special
tokens. All 320 ID sequences match the server's `/tokenize` response exactly.
Reuse API/engine 105110/105163, port 8010, without kernel changes or restart.
Prefix caching remains disabled and output remains one scored decision token.
Three repeat pairs alternate text/ID order, following an excluded warmup per
mode/concurrency. Frozen quick64 supplies c1; full320 supplies c128.

| Measurement | Text prompts | Exact token IDs | Observed change |
| --- | ---: | ---: | ---: |
| c1 median response latency | 164.82 ms | 159.47 ms | -5.35 ms (-3.25%) |
| c1 p95 response latency | 290.21 ms | 192.91 ms | -97.30 ms (-33.53%) |
| c1 pass throughput | 24,091 input tokens/s | 27,892 input tokens/s | +15.78% |
| c128 full-cohort throughput | 193,359 input tokens/s | 195,420 input tokens/s | +1.07% |
| c128 median response latency | 2.399 s | 2.412 s | +0.54% |

Table entries are medians of each pass's metric. Separately, pair each prompt's
repeat-median response latency: the median saving is 6.10 ms below 4k tokens,
37.08 ms at 4k--16k, and 95.18 ms at 16k+ (46/11/7 quick examples). These paired
statistics differ from subtracting two aggregate percentiles. The long-prompt
bin's median latency is 294.49 versus 194.41 ms.

The three c128 text rates are 191,856/197,727/193,359 tokens/s; IDs give
195,861/195,420/194,932. The small aggregate difference is within the observed
text-pass range, so this is not evidence of a robust new throughput gain.
At c128, API-process CPU totals fall from a median 7.34 to 2.83 CPU-seconds per
pass, while engine totals remain around 10 CPU-seconds. Process CPU sums include
all threads and can exceed elapsed wall time. The text frontend uses about
1.08 CPU cores on average in the median text pass; this is not whole-host CPU
saturation. There is substantial overlap with GPU work.

## Encoder versus the complete text path

Direct CPU encodes use the locked fast tokenizer, the same tokenizer files and
special-token policy, with every sequence checked against the frozen IDs.
Early three-repeat measurements span 0.78--1.23 ms overall median, with the next
set around 1.23--1.37 ms. A later matched call-path check gives:

| CPU call | Full320 serial pass | Overall median | p95 |
| --- | ---: | ---: | ---: |
| Plain AutoTokenizer | 3.268 s | 1.645 ms | 53.55 ms |
| vLLM CachedQwen2Tokenizer | 3.251 s | 1.623 ms | 53.49 ms |
| Cached tokenizer, completion length limit | 3.270 s | 1.645 ms | 53.89 ms |

The completion call uses the installed renderer's actual
`TokenizeParams(max_total_tokens=32768, max_output_tokens=1,
add_special_tokens=False)` encode kwargs: truncation enabled, max length 32768.
All results are fast-tokenizer sequences with exact parity; neither the cached
wrapper nor that limit establishes an extra slowdown. CUDA is hidden and remains
uninitialized in this CPU-only diagnostic. In the matched completion call,
median encode time is 1.32 ms below 4k tokens, 18.65 ms at 4k--16k, and 54.01 ms
at 16k+. Serial throughput is about 401k input tokens/s, still roughly twice
GPU serving throughput. CPU timing variation across checks limits precise
attribution; do not advertise the first, faster encode rate as a fixed capacity.

The server's serial `/tokenize` pass takes about 9.4 s for full320; round-trip
median is 11.4--11.6 ms and p95 is 112--117 ms. This includes HTTP parsing,
renderer/thread-pool work, ID-response serialization and client parsing. It is
not raw encoder timing or a clean estimate of inference tokenizer duration.

The text-versus-ID serving difference includes tokenization, frontend
validation/thread handoff, differing JSON payloads and resulting GPU batch
scheduling. Exact IDs are prepared outside timing. It is an ablation of the
text input path, not causal attribution of every saved millisecond to BPE.
Moving encoding to a caller still incurs that computation unless tokens are
already available or an exact reusable encoding can be cached. The earlier
7--8% GPU-window gaps remain engine-side observations, not tokenizer cost.

## Scores, failure receipt and collection

All c1 repeat-median scores and margins are exactly equal, with no threshold
flips or AUROC change. At c128, input IDs are still exact and arithmetic is
unchanged, but changed batch scheduling in this accepted FP4 stack changes
scores: mean/max difference 0.002087/0.104425, one threshold flip on an already
unstable text-control example. Source-macro/pooled AUROC deltas versus concurrent
text are +0.1580/+0.1328 percentage points. Do not interpret these as a quality
improvement or promote on this training-seen development population.

`tokenization01` preserves six completed c1 passes and an HTTP ReadError during
c128 warmup. The server remained healthy. The follow-up uses fresh connection
pools for each pass and reruns only c128 in `tokenization_batch02`; the transport
cause was not established and no server patch was made. `tokenization_callpath01`
contains the exact CPU script, nine timings and all parity checks.
`tokenization_collection01` binds logs, executed clients, pinned vLLM sources,
server/native receipts and **59 locally checksum-verified files**. Six focused
token-payload contract tests and Ruff pass. Selected baseline SHA256 remains
`39811c43e0b4bbf574e682d7b21f09e394909af3af4a69f3b398193cace89166`.
The sole healthy serving engine is retained; no long-running trial remains.

Subsequent work wires an optional
[native Gigatoken frontend](b200_gigatoken_frontend.md), with exact token IDs and
a substantial interactive tail-latency gain. Its c128 throughput is slightly
lower than its preceding HF text control; do not treat encoder speed as an
established GPU-throughput gain. Current retained API/engine is 106836/106874.
