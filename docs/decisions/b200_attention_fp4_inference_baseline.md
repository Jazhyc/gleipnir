# FP4 attention projections as the B200 inference reference

Decision date: 2026-10-06. The user says, “That's a pretty good throughput
increase. Let's make it the reference now.” Select the completed
`attention_fp4_serving01` recipe for subsequent B200 inference optimization.
This supersedes the [tuned FP4 reference](b200_tuned_fp4_inference_baseline.md).

Use `attention_fp4.json`: all 64 MLP, 48 large GDN and 16 full-attention
projection modules run native NVFP4 GEMMs with BF16 outputs. Retain combined
activation preparation, selected output tiles and adapted cuDNN MXFP8 causal
prefill. Small gates, recurrence, convolution, KV cache and decode remain BF16;
gates/state remain FP32. Preserve FP32 master adapters and reconstructable
packed serving weights. Experimental GEMM–SwiGLU fusion remains excluded.

## Evidence and explicit acceptance

Five warm c128 passes give a median **193,224 input tokens/s**, +5.77% against
the preceding reference. Aggregate throughput over all five passes is
193,243 tokens/s, +6.00%. Warm c1 median/p95 latency is **156.82/287.55 ms**,
6.68%/6.40% slower. Retain all initial sweep and confirmation measurements.

On the frozen 320-row training-seen systems-dev workload, repeat-median scores
give source-macro/pooled AUROC **0.878553/0.889729**, decreases of
**0.6055/1.8421 percentage points**. There are 20 threshold flips at 0.5.
Calibration and source-specific losses remain in the
[complete finding](../findings/b200_attention_projection_fp4.md); this selection
does not establish final-ID performance.

The new checksum-bound `quality_acceptance.json` records the user's selection
as `user_accepted_finite`. Strict master-score, inherited MXFP8 FP32-reference
and vendor preparation precision failures remain false. The new attention
projection arithmetic/isolation/changed-input replay receipt passes all 16
checks at the unchanged 1% decoded-reference ceiling. No tolerance changes.
Later changes still require finite scores, matched inputs, source bindings,
nonzero adapter effect and score/AUROC comparisons with this reference.

## Bound controls and resident worker

`experiments/b200_inference_benchmark/baseline.json` binds the original serving
summary, executed recipe copy, frozen workload, full sweep, warm throughput,
warm latency, native projection validation and quality acceptance by checksum.
Archive the preceding selection as
`baselines/frost_fp4_mlp_gdn_mxfp8_combined_prepare_shape_tuned.json`.
Use five warm confirmation passes for throughput and separate c1 confirmations
for latency; preserve original full-sweep score controls.

The entrypoint now defaults to `attention_fp4.json`. Selection changes no
kernel or capacity setting, so API/engine **93539/93625** stay resident on the
existing NC2 B200, port 8010. Stop serving before subsequent kernel/settings
changes or another GPU worker. Preserve shared caches and the existing pod.

## Next fusion opportunities

The earlier [GEMM–SwiGLU trial](../findings/b200_fp4_swiglu_fusion.md) reduced
the BF16 intermediate width and improved large-row producer timing by roughly
5%, but its serving results were mixed. Those results precede attention FP4.
The adapter still pads/copies partial row tiles, compiles exact-M plans and
writes BF16 activated values before a separate FP4 packer. Prioritize a
symbolic-M plan and removal of padding/copy overhead, then evaluate an output
quantization epilogue against the complete producer and down-projection path.

Direct FP4 output must account for the current whole-row amax over 9,216
activated columns. Independent GEMM column tiles cannot know that final row
scale without coordination. NVIDIA's current experimental
[grouped SwiGLU API](https://github.com/NVIDIA/cudnn-frontend/blob/develop/docs/fe-oss-apis/gemm_fusions/discrete_grouped_gemm_swiglu.md)
also lists only FP16/BF16/FP32 activated output for FP4 inputs; its quantized
output support is not a drop-in FP4-to-FP4 solution for this dense path.
Block-local or calibrated scales change arithmetic
and require a separate quality comparison; a smaller launch count alone is
insufficient evidence of a speedup. A fresh profile is needed before assigning
current bottleneck percentages. No additional fusion trial is launched by this
reference selection.

Superseded on 2026-10-07 by the user-selected
[direct FP4-output reference](b200_native_fp4_output_inference_baseline.md).
Preserve this decision and its original checksum-bound measurements.
