# Arithmetic-preserving serving optimizations for Qwen3.8-27B

Goal: speed up BF16 monitor serving of the trained Qwen3.8-27B
(`qwen38-27b-s0-01`, merged BF16) by porting the serving changes that do not
change precision. User scope, 2026-10-18: BF16-preserving changes only; the
FP8 attention / FP4 MLP and GDN projection recipe stays reserved for a future
QAD-adapted 27B ([serving decision](../../docs/decisions/b200_monitor_score_reference.md),
[transfer note](../../docs/research/qwen38_27b_serving_transfer.md)).

**Candidates**, each validated on the 4B as bitwise or exact-arithmetic:
1. last-query final-layer attention (layer 63 attends only from each
   request's pooled last token; every token still writes K/V);
2. the GDN context-parallel route gate (calls under 8,192 tokens take the
   non-CP route);
3. the lean GDN prefill host path (bitwise identical to upstream GDN).

**Control.** The campaign's BF16 server (`bf16_server`, `Bf16Worker`) on the
merged trained 27B, with the same scheduler, chunked prefill, prefix caching
off, 32,768 context and runner capacity 128. First the backlog's
batch/singleton consistency gate for BF16.

**Measurements** (per the inference guide): prompt tokens/s as the main
throughput comparison, plus requests/s and latency, on a fixed workload and
concurrency with all repeats; per-source and macro AUROC deviation and score
parity against the archived BF16 control predictions.

**Acceptance (frozen before measuring).** A candidate is kept when its scores
are bitwise equal to the control, or for the route gate within the control's
own repeat variation with no AUROC change beyond repeat noise; when every
startup check shows the layer actually ran; and when prompt-token throughput
improves by at least 2%. Adoption into a 27B serving default is the user's
decision.

**Stop** on missing or nonfinite scores, a failed startup check, failed
parity, or a mismatch in prompt identity or token counts.

## Run log

**Harness.** `src/gleipnir/serving/bf16_launch.py` starts the campaign's BF16
server for any merged model (selected command → merged model →
`bf16_monitor.command` → worker class); `bench.py run` warms up on quick64,
times full320 (1,310,581 prompt tokens) at concurrency 128 three times and
quick64 at concurrency 1, checks the BF16 audit and retires the server;
`bench.py compare` reports throughput ratios, score parity and AUROC deviation.
The workload uses the 4B judge prompt, so AUROC here is a systems diagnostic.
`src/gleipnir/serving/bf16_stack.py` holds config-derived copies of the
last-query and fast-GDN mixins: the FP8 default's worker sources are hash-bound
in `serving_default.json`, so they are not edited.

**Control01 (2026-10-18 01:30–01:35 UTC, `GPU-7a2d0c53…`).** BF16 audit
passed (16 attention layers). c128 full320: 22,874 / 22,990 / 22,943 prompt
tokens/s (5.59–5.61 requests/s; spread 0.5%). c1 quick64: 12.9 s. Engine
startup about 60 s.

**BF16 last-query canary** (`last_query_bf16.py`,
`results/qwen38_27b_serving/last_query_bf16.json`). All 8 cases bitwise equal
on stock FlashInfer 0.7.0.post1 TRTLLM prefill at the 27B attention shape (24
query / 4 KV heads, head dim 256, page 16): 32K first/second chunks, 1M
history, irregular split, mixed prefills and a 128-request batch, with
every-other-request subsets.

**Candidates (2026-10-18 01:35–01:49 UTC; same GPU, harness and workload).**

| condition | c128 prompt tok/s (3 repeats) | vs control | c1 p50 | score MAE vs control | AUROC Δ macro / max source | 0.5 flips |
|---|---|---:|---:|---:|---:|---:|
| control01 | 22,874 / 22,990 / 22,943 | | 38.3 ms | | | |
| fastgdn01 | 23,397 / 23,410 / 23,372 | +1.99% | 35.8 ms | 0.00034 | 0 / 0 | 1 |
| lastquery01 | 23,063 / 23,009 / 23,009 | +0.40% | 39.1 ms | 0.00018 | 0 / 0 | 0 |
| **lqfast01** | 23,413 / 23,510 / 23,453 | **+2.28%** | **35.1 ms** | 0.00008 | 0 / 0 | 0 |

- Repeat-to-repeat noise at c128 (pairwise within the control): score MAE
  0.00026–0.00065, maximum 0.031, margin 0.125 (one ulp); 298–304 of 320
  rows identical. Batch composition changes BF16 kernel choices, so separate
  servers cannot show bitwise equality; every candidate is inside this
  envelope. The fast-GDN flip is a row already unstable across control
  repeats.
- Each layer ran: fast GDN installed on all 48 GDN layers; last-query
  attended on `layers.63` (call receipt).
- Fast GDN alone is +1.99%, just under the frozen 2% bar; last-query alone
  (+0.40%) is small, as expected for one of 64 layers. Together they pass
  (+2.28% throughput, −8.3% c1 latency).

**GDN route gate (`route_sweep.py`, `route01`, 01:52–01:57 UTC).** In-server
paired A/B at C1 (BF16 + last-query + fast GDN; 3 alternating rounds):
FlashInfer's `auto` route versus non-CP for every chunk. Median p50: 1K
49.2 vs 49.2 ms, 4K 181.9 vs 182.4, 8K 357.6 vs 357.2, 16K 725.0 vs 722.8
(non-CP faster 3/3, −0.3%), 32K 1,510.0 vs 1,510.9, quick64 33.3 vs 36.1.
Scores stable within each arm; the audit shows both routes ran (22,656 auto,
22,608 non-CP calls). No length shows a meaningful difference, so the gate,
which is not bitwise, is rejected for the 27B: its 4B gain came from CP setup
overhead that is negligible next to the 27B's GDN work.

## Result and decision

User decision, 2026-10-18: **regular evaluations of every model (4B and 27B)
use stock vLLM BF16** (`Bf16Worker`). All serving optimizations belong to the
optimized path, together with quantization, for QAD-adapted deployment
candidates, so the final comparison is stock BF16 against fully optimized QAD.
The arithmetic-preserving stack is therefore not a separate evaluation mode.

What these measurements establish for a future optimized 27B path:
- **Carry over:** last-query final-layer attention and the lean GDN host path
  (together +2.28% c128 prompt throughput, −8.3% c1 latency at BF16, scores
  within repeat noise). Their config-derived mixins are in
  `src/gleipnir/serving/bf16_stack.py`; the BF16 last-query receipt covers the
  27B attention shape.
- **Leave out:** the GDN route gate (no gain on the 27B at any length).
- The 4B optimized default keeps its own hash-bound workers; the generic
  mixins are building blocks for the 27B path, not a change to the 4B default.

Arithmetic-preserving changes give only a few percent on the 27B because BF16
GEMMs dominate its serving time; the larger lever is the precision recipe,
which waits for a QAD-adapted 27B.

## Quantized stack, phase 1 (2026-10-18)

User request: complete the optimized 27B stack, in phases; drift is reported
against BF16, not gated (no QAD-adapted 27B exists, so this is not a
deployment selection).

**Phase 1** (`src/gleipnir/serving/quant_stack.py`, quantization
`gleipnir_geometry_fp8_fp4`, server `quant_server.py`): the unchanged FROST FP4
MLP and GDN methods; FP8 per-token/per-channel CUTLASS attention projections
checked against config-derived shapes (27B qkv 14336×5120 with the output
gate, o 5120×6144; `test_quant_geometry.py` also pins the 4B shapes); BF16
FlashInfer attention with last-query (the BF16 receipt still applies) and the
lean GDN path. `QuantAuditWorker` counts coverage from `layer_types`. The 4B
default's hash-bound modules are reused by import only.

**quant01** (02:26–02:30 UTC, same harness and GPU). Load audit: 32 FP8
attention, 128 FP4 MLP, 96 FP4 GDN projections (97 small linears stay BF16);
FROST layer-0 reference self-checks passed; last-query on layer 63.

| | control01 (BF16) | quant01 |
|---|---:|---:|
| c128 prompt tok/s (3 repeats) | 22,874 / 22,990 / 22,943 | 47,757 / 47,524 / 47,614 (**2.08×**) |
| c128 requests/s | 5.60 | 11.63 |
| c1 p50 latency | 38.3 ms | 51.0 ms (+33%) |
| score MAE / max vs BF16 (diagnostic workload) | | 0.023 / 0.43 |
| macro AUROC Δ (diagnostic workload) | | −0.30 pt |

Throughput doubles; short-request latency worsens because the unfused FP4 path
packs activations separately and uses untuned small-M plans. That is what the
deferred phase-2 pieces address: fused FP4 producers, native SwiGLU output
(needs a new tile: N192 does not divide 34,816 and its stores do not guard
columns), tuned GEMM tiles, and MXFP8 attention (24/4 heads never validated).

**ID drift (`id_drift.py`, `quantid01`, 02:31–02:44 UTC).** All 3,012 frozen
ID rows (the campaign's student-prompt workload) through the phase-1 stack at
c128, against the campaign's BF16 ID predictions:

| ID | BF16 pAUROC@20 / AUROC | phase 1 pAUROC@20 / AUROC |
|---|---:|---:|
| macro | 0.9217 / 0.9822 | **0.9200 / 0.9818** |
| Gloom exfiltration | 0.8868 / 0.9742 | 0.8848 / 0.9734 |
| Test STRIDE | 0.9567 / 0.9901 | 0.9552 / 0.9903 |

Score correlation 0.995 (MAE 0.022, maximum 0.22); 59 of 3,012 rows cross
0.5. ID throughput 48,827 prompt tokens/s (691 s for all rows, against 1,456 s
at BF16 in the campaign). Without QAD the quantized 27B loses 0.17 ID points;
this is reported, not gated, per the user. A QAD-adapted 27B would be the
deployment candidate.

**c1 latency by length (phase 1 vs BF16).** The quantized path is about 2×
faster from ~1.5K tokens up (4,856 tokens: 108 vs 219 ms; 14,373: 315 vs 650
ms) but has a flat ~49 ms floor from ~260 to ~1,400 tokens (BF16 27–74 ms),
while a 188-token request takes 19 ms. A length-independent floor points to
host/launch overhead; the working hypothesis (unverified) is that very short
steps run inside vLLM CUDA graphs while 260+ tokens run outside them, where
each of the 240 FP4 projections adds a Python custom-op call and separate
activation-packing launches. A host profile (`GLEIPNIR_HOST_PROFILE_DIR`) of
a ~500-token request would confirm it; fused producers and graph capture for
these sizes are the fixes. Long-context throughput additionally needs MXFP8
attention: phase 1 quantizes only projections (linear in tokens), so at
~1M-token contexts the BF16 attention layers dominate and the speedup shrinks.

## Quantized stack, phase 2 (2026-10-18, second Pod)

Same user scope (drift reported, not gated). New EU-RO-1 B200 Pod
`wdbgkrnymh3n8h` (`GPU-4b31de30…`, driver 580.178.04); every condition below
reran on this GPU. `remerge.py` rebuilt the merged checkpoint byte-identical
to the campaign merge (all source and output hashes, 219 s), so the
campaign's parity and ID receipts apply.

**Ported modules** (new geometry modules; the 4B default's hash-bound sources
are reused by import only):
- `mxfp8_geometry.py` + `Mxfp8AttentionMixin`: the 4B MXFP8 producers with a
  cuDNN plan and guards for the served GQA heads, installed ahead of the BF16
  attention audit (which would otherwise restore BF16 after its count).
- `fp4_geometry_tuning.py` + `TunedGemmMixin`: the four FROST FP4 shapes from
  the config (27B: gate_up 5120×34816, GDN in 5120×16384, down 17408×5120,
  GDN out 6144×5120; the derivation reproduces the 4B table exactly).
- `fp4_geometry_producers.py` + `ProducerFusionMixin`: the 4B fused
  residual-RMSNorm and SwiGLU FP4 producers and the vendor GDN packer with
  K-checked wrappers, own op names and config-derived install scope.
- `QuantFullWorker` stacks all of these with last-query and lean GDN.

**Receipts** (`results/qwen38_27b_serving/`):
- `mxfp8_canary01.json` (24/4 heads): 7/7 within 0.0015–0.0017 of the FP32
  MXFP8 oracle (limit 0.01); bitwise equal to the 4B module at 16/4. Relative
  to full BF16 attention 0.040–0.053, 4 of 7 over the strict 0.05, as in the
  4B receipt (accepted there as finite only); recorded, not gated.
- `last_query_mxfp8.json`: 8/8 bitwise, including the 1M-token history.
- `fp4_gemm_tune01.json`: all tiles pass; only the two down projections gain
  at M ≤ 4096 (128×128 cluster 2×1, +5–6%); the base tile elsewhere.
- `fp4_producers01.json`: all checks within 0.006 decoded / 0.0003 GEMM L2,
  exact residuals. SwiGLU (8 warps) 1.0×/1.6×/2.3× at 1/129/32768 rows,
  norm (4 warps) 2.0×/2.5×/3.3×; vendor 0.5× at 1 row, 2.2× at 32768. The
  feared K17408 spills did not appear.

**Bench** (full320 c128 × 3, quick64 c1; scores on the training-seen
diagnostic workload):

| condition | c128 prompt tok/s | vs BF16 | macro AUROC Δ | 0.5 flips |
|---|---|---:|---:|---:|
| control02 (BF16) | 23,100 / 23,093 / 23,075 | 1.00× | | |
| quant02 (phase 1) | 47,905 / 47,889 / 47,775 | 2.07× | −0.30 pt | 8 |
| mxfp801 (+ MXFP8 attention) | 48,074 / 50,111 / 50,024 | 2.14× | +0.37 pt | 9 |
| tuned01 (+ tuned tiles) | 48,459 / 50,117 / 50,062 | 2.15× | +0.37 pt | 9 |
| **full01** (+ fused producers) | 52,306 / 53,088 / 53,052 | **2.29×** | −1.11 pt | 8 |
| graphs01 (full, graphs ≤ 2,048) | 52,069 / 53,126 / 53,064 | 2.28× | −1.11 pt | 8 |

Every live audit passed (MXFP8 on all 16 attention layers; tuned tiles on all
four shapes; norm, SwiGLU and vendor producers observed).

**ID drift (`fullid01`, all 3,012 frozen rows, c128).** Macro pAUROC@20
0.9188 / AUROC 0.9816 (BF16 0.9217 / 0.9822; phase 1 0.9200 / 0.9818). Gloom
exfiltration 0.8824 (BF16 0.8868), STRIDE 0.9552 (0.9567). Score MAE 0.0215,
correlation 0.9952, 57 rows cross 0.5 (phase 1: 0.0222, 0.9950, 59). 54,238
prompt tokens/s, 622 s for all rows (BF16 campaign 1,456 s: 2.34×). The
shutdown `EngineDeadError` in the log follows the harness SIGTERM after all
3,012 requests returned.

**c1 latency floor.** Median latency by prompt length (quick64; ms; full02
and graphs01 are medians of 3 passes, the others 1):

| prompt tokens (rows) | BF16 | phase 1 | full02 | **graphs01** |
|---|---:|---:|---:|---:|
| < 256 (8) | 26.3 | 19.5 | 17.8 | 19.8 |
| 256–1,500 (36) | 34.0 | 50.5 | 54.8 | **22.4** |
| 1,500–4,000 (2) | 127.6 | 68.0 | 66.1 | 59.9 |
| 4,000–10,000 (8) | 279.3 | 133.5 | 123.6 | 122.7 |
| 10,000+ (10) | 822.6 | 409.2 | 369.5 | 369.1 |

vLLM captures piecewise graphs only up to 256 tokens; above that the
quantized stack's ~5× more custom ops per layer run eagerly from Python, and
each added layer made the floor worse (phase 1 50.5 → full 54.8 ms). Extending
capture to 2,048 tokens (`--compilation`; 47 sizes, padding ≤ 25%, all inside
the M ≤ 4096 tile band) removes it: 22.4 ms, faster than BF16, with c128
unchanged, 7 s capture and 0.9 GiB graph memory. This confirms the host
overhead hypothesis above. Padding shifts c1 scores by 0.005 mean / 0.083 max
against full02, comparable to batch-composition noise.

**Native SwiGLU FP4 output** (user request, same day).
`fp4_geometry_swiglu.py` reuses the 4B source transform and weight
interleaving, substitutes the 27B output width in the direct stores, and
compiles a tile that divides the gate/up width (the 4B's N192 does not divide
34,816 and its stores guard rows only). `fp4_swiglu_native02.json`: both
256×256 and 256×128 compile and pass every check on all nine row counts
(packing equals the independent E4M3/E2M1 reference exactly; arithmetic and
changed replay ≤ 0.003; rows isolated); the generated source at the 4B width
equals the 4B module's. Activation error against BF16 is 0.15–0.21 relative L2
(4B receipt: 0.07–0.09), consistent with more 16-value blocks of the ×20
outlier row saturating at the unit-global-scale ceiling (448 × 6); unverified.
Full-MLP speed with 256×128 is 1.04–1.13× at 17–4,096 rows and 0.96–0.99× at
29K–32K (the reverse of the 4B); the frozen rule (longest contiguous ≥ 2% run)
selected rows 17–1,536.

`native01` (full stack + native output + graphs to 2,048; native on all 64
MLPs): c128 52,309 / 53,092 / 52,388 (2.28×, same as graphs01); c1 medians
22.9 / 24.0 / 63.3 / 130.0 / 375.2 ms across the bands above, slightly slower
than graphs01 in every band; c1 scores drift 0.013 mean / 0.132 max against
full02. ID (`nativeid01`): pAUROC@20 0.9192 / AUROC 0.9817, MAE 0.0215, 58
flips, 53,775 prompt tokens/s (full01: 0.9188, 54,238). Most ID prompts exceed
the band, so this barely exercises the native arithmetic. **Rejected for the
27B:** the operator gain does not reach the served model (inside graph-captured
steps the MLP is a small share and the dispatch adds an op), and it changes
activation quantization. The code and receipt are kept.

**Status.** Recommended optimized 27B configuration: `QuantFullWorker`
(MXFP8 attention, tuned FP4 tiles, fused producers, last-query, lean GDN)
with piecewise graphs to 2,048 tokens (`graphs01`): 2.28× BF16 at c128, about
2.2× on long prompts at c1, faster than BF16 on short prompts, at −0.29 ID
pAUROC@20 points without QAD. Long-context (> 32K) throughput with MXFP8 is
not measured here.

## Long context, 1K–1M tokens (2026-10-18)

User request: latency and throughput of the recommended optimized stack
against stock BF16 from 1K to 1M tokens. `long_context.py` (`long01`):
concurrency 1, the deterministic unlabelled synthetic prompts of
`b200_long_context` (exact token counts), one excluded warmup then 5 requests
(3 from 128K). As in `b200_context_million`: native RoPE to the checkpoint's
262,144 positions, then YaRN ×4 (`--max-model-len 1048576`) for 512K and 1M,
with a 256K overlap. Unchanged 32K chunk budget, 128-sequence cap and
scheduler; the optimized stack is `QuantFullWorker` with graphs to 2,048.

MXFP8 beyond 32K needs a raised descriptor/producer bound;
`mxfp8_envelope.py` ports the 4B 1M envelope check to 24/4 heads
(`mxfp8_envelope01.json`): short and batched outputs bitwise unchanged after
raising the bound; 32K query chunks against 64K–1M histories (pages beyond
2^31 cache elements) and a 32K+17 history within 0.0017 of the quantized
oracle (limit 0.01); 0.054–0.056 from full BF16 (recorded, as for the 4B).
`Mxfp8AttentionMixin` raises the bound only with this hash-bound receipt.

| prompt tokens | BF16 median (range) | optimized median (range) | speedup | optimized prompt tok/s |
|---:|---:|---:|---:|---:|
| 1K | 0.056 s (0.053–0.057) | 0.029 s (0.029–0.034) | 1.95× | 35,573 |
| 2K | 0.094 s | 0.046 s | 2.05× | 44,693 |
| 4K | 0.186 s | 0.082 s | 2.27× | 49,998 |
| 8K | 0.363 s | 0.159 s | 2.29× | 51,652 |
| 16K | 0.730 s | 0.330 s | 2.21× | 49,700 |
| 32K | 1.533 s | 0.697 s | 2.20× | 46,989 |
| 64K | 3.395 s | 1.575 s | 2.16× | 41,613 |
| 128K | 8.537 s | 3.940 s | 2.17× | 33,270 |
| 256K | 24.99 s | 11.04 s | 2.26× | 23,751 |
| 256K (YaRN) | 25.09 s | 11.07 s | 2.27× | 23,672 |
| 512K (YaRN) | 82.86 s | 35.41 s | 2.34× | 14,806 |
| 1M (YaRN) | 298.5 s (296.6–302.8) | 128.6 s (128.3–128.6) | **2.32×** | 8,156 |

- The speedup holds from 4K to 1M (2.2–2.3×) and grows slightly where
  attention dominates (512K–1M): MXFP8 prefill attention keeps pace with the
  FP8/FP4 projections, so the throughput gain does not collapse at long
  context. Without it (phase 1), attention would run in BF16 and the speedup
  would shrink toward 1× as attention's share grows; that arm was not run.
- Latency grows roughly quadratically past 64K (each doubling ×2.4–3.6), as
  attention takes over. At 1M BF16 runs at 3,512 prompt tokens/s, the
  optimized stack at 8,156.
- YaRN does not change timing (256K: 25.09 vs 24.99 s BF16, 11.07 vs 11.04 s
  optimized). Startup 92–100 s; GPU memory in use 160–170 GB for both.
- Scores repeat exactly within each stack; they differ between stacks by up
  to 0.14 on these repetitive unlabelled prompts and beyond 256K are YaRN
  outputs, so they are not quality evidence.
