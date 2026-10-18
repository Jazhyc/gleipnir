# Transferring the 4B serving optimizations to Qwen3.8-27B

Status: arithmetic-preserving part done 2026-10-18
([experiment](../../experiments/qwen38_27b_serving/README.md)): last-query
attention + the lean GDN host path give +2.3% BF16 c128 prompt throughput and
−8.3% c1 latency with no score change; the GDN route gate brings nothing on
the 27B. Per the user (2026-10-18) regular evaluations stay on stock BF16, and
these layers join the FP8/FP4 recipe (steps 3–4 below) in the optimized path
for a QAD-adapted 27B.

**Quantized phase 1 done (2026-10-18):** FP8 attention projections + FROST FP4
MLP/GDN with config-derived geometry (`src/gleipnir/serving/quant_stack.py`),
BF16 attention, last-query and lean GDN: 2.08× BF16 c128 throughput, ID
pAUROC@20 0.9200 vs 0.9217 at BF16 (no QAD). c1 latency is 33% worse on short
prompts. **Phase 2 mostly done (2026-10-18):** MXFP8 attention at 24/4 heads, a 27B
FP4 tile table, fused norm/SwiGLU/vendor FP4 producers and CUDA graphs up to
2,048 tokens: 2.28× BF16 c128 throughput, ID pAUROC@20 0.9188 vs 0.9217 at BF16
(no QAD), and the short-prompt latency floor gone (256–1,500 tokens: 22 ms vs
34 ms at BF16). Open: native SwiGLU FP4 output (N256 tile, 27B store
constants) and a long-context measurement. Build these as new geometry
modules; the 4B default's sources are hash-bound
([port plan and results](../../experiments/qwen38_27b_serving/README.md)).

**Why.** Unadapted Qwen3.8-27B reaches 0.890 Mean-OOD pAUROC@20 at the
shared 27B cost proxy ($1.85/1K), on the OOD frontier and 1.8 points below
Kimi K3 ([finding](../findings/qwen27b_generations_ood.md),
[frontier](tool_trajectory_ood_frontier.md)). Its OOD and ID runs used plain
BF16 vLLM 0.31 with none of the 4B serving optimizations
([serving decision](../decisions/b200_monitor_score_reference.md)).

## Architecture (from the pinned configs)

| | Qwen3.5-4B | Qwen3.8-27B (`1d4bf0f2…`) |
|---|---:|---:|
| layers (full attention / GDN) | 32 (8 / 24) | 64 (16 / 48) |
| hidden / MLP width | 2,560 / 9,216 | 5,120 / 17,408 |
| attention heads (query / KV), head dim | 16 / 4, 256 | 24 / 4, 256 |
| GDN heads (key / value), head dims | 16 / 32, 128 | 16 / 48, 128 |
| final layer | full attention | full attention |
| tied embeddings | yes | no |

Both models use the same layer types in the same pattern (GDN, with full
attention every fourth layer); the 27B is wider and deeper.

## Expected transfer

**Should carry over directly:**
- merged BF16 serving on vLLM 0.31 with a persistent engine (already used);
- last-query final-layer attention: layer 63 is full attention. The relative
  saving is about half the 4B's, since it covers 1 of 64 layers;
- the GDN host-path and context-parallel routing changes: host-side and
  shape-agnostic, but worth less because per-token GPU work is about 7× larger.

**Needs adaptation:**
- the FP8 attention / FP4 MLP and GDN projection recipe. It hardcodes 4B
  shapes: `gleipnir.serving.gdn_precision.SHAPES` is `(2560, 12288)` /
  `(4096, 2560)`, and the attention worker audit expects exactly 16 FP8
  projections (the 27B has 32). The FP4 GEMM tuning used 4B shapes. This is the
  likely largest speedup, because the 27B GEMMs are larger and more
  compute-bound, and it roughly halves weight memory.

**Needs fresh validation:**
- low-precision score quality: the 4B acceptance does not extend to the 27B;
- the batch/singleton consistency gate: the earlier FP8 Qwen3.8 checkpoint
  failed it on vLLM 0.24 ([ID screen](../../experiments/qwen122b_id/README.md)),
  and it has not been run on BF16.

## Plan

1. Run the BF16 batch/singleton consistency gate (cheap; do first).
2. Enable last-query attention and the GDN host changes, with their bitwise or
   near-bitwise checks.
3. Generalize the FP8/FP4 shape bindings and projection counts; re-tune FP4
   GEMMs at 27B shapes.
4. Before adoption, measure speed and score change against BF16 on ID and OOD,
   with the same parity gates the 4B recipe used.
