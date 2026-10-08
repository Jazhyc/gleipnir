# Full-attention projection precision recovery

Hypothesis: replacing only the eight full-attention blocks' 16 FP4 QKV/output
projections with FP8 W8A8 recovers part of the earlier pAUROC@20 loss at a
smaller throughput cost than BF16. BF16 supplies a projection-only recovery
bound. The user authorizes all three conditions on the existing B200.

Freeze vLLM 0.31, the completed 272-update FP4/FA4-trained 4B adapter and its
FP32-accumulated/BF16-exported merged checkpoint. Keep all 64 FP4 MLP and 48
FP4 GDN projections, native SwiGLU output, MXFP8 attention core, BF16 cache and
recurrence, FP32 gates/state, CUDA GDN with automatic context parallelism,
native Gigatoken/direct FROST, causal LAST two-row BF16 head, prefix caching off,
stock synchronous FCFS and 32K context/chunk/128-sequence limits fixed.

Reuse archived `b200_vllm031/diagnostic_default02` FP4 development repeats and
`b200_vllm031/id02` FP4 ID predictions; do not rerun an unchanged control.
New BF16/FP8 projection operators use opaque native linear calls so auditing
does not enter compiled graphs. FP8 has per-output-channel weight scales and
dynamic per-token activation scales, native CUTLASS GEMM and BF16 outputs.
Quantize original merged BF16 weights, never decode existing FP4 weights.

Before serving, validate both geometries (K2560/N10240 and K4096/N2560) at
rows 1/17/129/1536/2304/4096/29184/32768: finite outputs, decoded quantized
FP32 reference error <=1%, exact zero rows and untouched-row isolation, changed
input CUDA-graph replay, and BF16-relative error recorded separately. Require
all 16 actual model projections and unchanged non-target precision/adapter
identity. Record strict master/serving score limits (MAE 0.02/0.005 and
correlation 0.99/0.995), never convert a failure into parity. User-authorized
precision-recovery comparisons may continue past finite score disagreement,
with exact tokens and a nonzero adapter effect; native/provenance failures stop.

For each new condition, exclude warmup and run quick64/c1 three times and
full320/c128 six times, matching the archived workload and HTTP protocol.
Report all repeats, prompt tokens/s, requests/s, latency bins, pooled/per-source/
source-macro pAUROC@20 and AUROC, calibration, ties and fixed-0.5 diagnostics.
Then run one full ordered 128-row-group pass on all 3,012 canonical CoT-removed
ID examples for **both predeclared candidates**, regardless of development
ranking. ID is a descriptive fixed-checkpoint comparison, never a search loop.
Compare with archived 0.31 FP4, 0.24 optimized and same-adapter BF16 predictions.
No OOD, threshold fitting, checkpoint selection or automatic promotion.

Screen preference is frozen: retain candidates improving development source-macro
pAUROC@20 by >=0.005 with no pooled decline; among those within 0.005 of BF16
macro pAUROC prefer the fastest, otherwise retain BF16 as the measured quality
bound. This is diagnostic triage only; the mixed cohort has partial training
overlap and cannot establish held-out parity or select production precision.
Always report both ID candidates; do not change or rerun them based on ID.

Stop on provenance drift, unsupported native dispatch, truncation, missing/
nonfinite outputs, zero adapter effect, server/transport failure, OOM or suite
completion. Preserve failed receipts and partial passes. Replace incompatible
servers sequentially, retain shared caches and collect artifacts before retiring
only the owned process. Do not create or terminate capacity. No in-chat scheduling
tool is available; startup and completion are monitored during the active turn.

```bash
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_attention_precision.native --name native01
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_attention_precision.run --name precision01 \
  --native-receipt results/b200_attention_precision/native01.json
```
