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

## Result

Recommended BF16 27B serving stack: **last-query final-layer attention + the
lean GDN host path** (`Bf16LastQueryFastGdnWorker` with the BF16 last-query
receipt): +2.28% c128 prompt-token throughput and −8.3% c1 median latency,
scores within control repeat noise and no AUROC change. Arithmetic-preserving
changes give only a few percent on the 27B because its serving time is
dominated by BF16 GEMMs; the larger levers are the precision recipe (reserved
for a QAD-adapted 27B) and the opt-in prefix-cached mode
([decision](../../docs/decisions/prefix_cached_serving.md)). Adoption as the
27B BF16 serving path is the user's decision.
