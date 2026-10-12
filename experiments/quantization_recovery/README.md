# Quantization-aware monitor recovery

Test whether short same-model distillation recovers the strict-filtered 4B
monitor's optimized-serving ranking. The completed BF16 monitor is the frozen
teacher; the student starts from its FP32 rank-128/alpha-256 adapter. This is
recovery of deployment precision, rather than another Kimi distillation or
partial-AUROC intervention.

Use the unchanged 7,836 strict-filtered training rows and their original prompt
bytes, pinned Qwen3.5-4B revision, seed 0, binary KL at temperature 1, AdamW
1e-5, zero weight decay, 3% warmup and linear decay for one epoch (245 updates).
Logical batch remains 32, complete-example packing budget 16,384 tokens and
context cap 29,696. Cache the teacher's two raw decision logits and normalized
distribution with prompt/model/merge hashes; retain source labels separately.
There is no hard-label loss, original-teacher mixture or held-out selection.

Quantize effective BF16 merged weights after FP32 adapter accumulation, grouping
gate/up and GDN QKV/Z as serving does. Use native NVFP4 projections and the
serving SwiGLU row policy, per-channel/per-token FP8 full-attention projections,
and the repaired serving MXFP8 attention forward, including FP8 probabilities.
Keep BF16 recurrence and remaining operators. Backward uses a straight-through
low-rank approximation with BF16 contractions and FP32 adapter gradients;
attention uses a quantized-QKV FA4 surrogate, not a claimed exact derivative of
the serving kernel. MLP backward reconstructs gate/up activations from retained
FP4 operands locally; the training shell is eager, without model checkpointing.
HF normalization/recurrence may differ from vLLM, so an
actual model-score alignment gate is required before training.

First validate operators, merged-weight identity, sequence isolation, finite
adapter gradients and longest-row memory. Compare the untrained simulator with
saved optimized-server canary predictions (MAE <=0.02, correlation >=0.99).
Preserve failed receipts and stop on drift, nonfinite/missing gradients, OOM,
failed alignment or incomplete coverage. Do not reuse the old frozen-base FP4
receipt to authorize this new arithmetic. A failed gate is an experimental
result and prevents full training.

Select only the final epoch. Merge into one reconstructable ephemeral checkpoint,
preserving FP32 masters and merge receipts on the existing NC2 network volume.
Before scaling evaluation require adapter-specific master/serving parity.
Evaluate the frozen 3,012-row CoT-removed ID set with two optimized c128 passes;
compare cached BF16 and optimized baselines. Report source-macro raw normalized
pAUROC@20, pooled/per-source AUROC, calibration, threshold diagnostics, ties,
repeat score differences and input tokens/s. No OOD evaluation or automatic
promotion. Teacher scoring is an explicit BF16 exception to serving defaults.

`python -m experiments.quantization_recovery.run` orchestrates preparation and
gated stages. Outputs are under `results/quantization-recovery01/`, teacher cache
under its `data/` tree and logs under `logs/runpod/quantization-recovery01/`.
Use the existing NC2 B200 and shared persistent compiler caches. No in-chat
scheduler is available; detached automation does not promise later agent checks.
