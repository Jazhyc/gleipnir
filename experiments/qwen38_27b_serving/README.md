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
