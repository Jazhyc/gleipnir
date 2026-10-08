# FP8 GDN projections on the FP8-attention default

Hypothesis: replacing the 24 GDN blocks' 48 FP4 QKV/Z and output projections
with native FP8 W8A8 recovers low-FPR ranking at a modest throughput cost.
One predeclared FP8 candidate; no automatic promotion or ID-driven search.

Freeze the current vLLM 0.31 FP8 full-attention default, final 272-update
FP4/FA4-trained 4B adapter and original merged BF16 weights. Keep all 64 FP4
MLP projections and native SwiGLU output, MXFP8 attention core, small BF16
GDN gates/convolution/recurrence operands and FP32 gates/state unchanged.
Preserve native Gigatoken/direct FROST, causal LAST two-logit scoring, stock
synchronous FCFS, prefix caching off, CUDA GDN/automatic context parallelism,
32K context/chunk and 128 sequence limits. Quantize BF16 weights directly.

Baseline: archived `b200_attention_precision/precision01/fp8` development
repeats and ID predictions, verified against the current selection. No unchanged
control rerun. Native checks cover both GDN geometries K2560/N12288 and
K4096/N2560 at rows 1/17/129/1536/2304/4096/29184/32768, quantized-reference
relative L2 <=1%, finite outputs, exact zero/untouched rows and changed-input
CUDA-graph replay. Record BF16-relative error separately. Audit all 48 actual
FP8 calls and the unchanged attention/MLP/head/core identities.

After provenance/native admission, run the 20-row canary, three quick64/c1
and six full320/c128 repeats with excluded warmups and fresh HTTP pools, then
one full 3,012-row CoT-removed ID pass in original ordered 128-row partitions.
The user-requested precision diagnostic may continue past finite score
disagreement with a nonzero adapter effect; preserve strict failed flags.
Report score drift, pooled/per-source/source-macro AUROC and raw normalized
pAUROC@20, calibration, fixed-0.5 diagnostics, ties, all timing repeats,
input tokens/s, requests/s and latency. ID remains descriptive; no threshold,
checkpoint or precision selection, OOD evaluation or automatic default switch.
Development diagnostic preference requires macro pAUROC gain >=0.005 with
no pooled decline; report speed costs independently without a promotion rule.

Stop on source/provenance drift, unsupported native dispatch, missing/nonfinite
scores, truncation, zero adapter effect, transport/startup failure, OOM or suite
completion. Retire only the owned candidate process, preserve partial/failed
receipts, capacity and shared caches. No scheduling tool is available; monitoring
continues during the active turn, without promised follow-ups after it ends.

```bash
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_gdn_precision.native --name native01
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_gdn_precision.run --name gdn01 \
  --native-receipt results/b200_gdn_precision/native01.json
```
