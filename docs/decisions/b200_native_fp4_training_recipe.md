# Native FP4 MLP B200 training default

On 2026-10-05 the user explicitly selected “Make the FP4 baseline the default”.
`qwen35_4b_b200_default` now aliases `qwen35_4b_b200_fp4_mlp`. Preserve the
previous BF16 FA4 recipe as `qwen35_4b_b200_bf16_fa4`; historical benchmark
runners explicitly use it. This supersedes the BF16 default selection without
changing completed campaign receipts or establishing held-out quality equivalence.

All 32 MLPs use native NVFP4 frozen contractions in forward and base input
gradients, row-scaled hardware activation packing, 16x16 weight scales, merged
gate/up contractions and fused descale. Original BF16 frozen parameters and
FP32 LoRA masters/state-dict names remain intact. Ordinary LoRA contractions
retain their validated precision. All 24 GDN layers keep BF16 Q/K/V and original
projections/convolution/normalization with pinned FlashQLA and FP32 gates/norm.
Full attention uses BF16 causal varlen FA4 4.0.0b33. Retain rank 128/alpha 256,
no dropout/checkpointing, logical batch 32/accumulation 1, 16,384 packed tokens
and context cap 29,696. Compile the full-attention/linear shells using Inductor
default mode with dynamic shapes; mark each physical model forward as a CUDA
graph step boundary. Exclude all rejected GDN/convolution/dispatch/normalization/
grouped-Q/K candidates. Latest pooled resident speed is 3.67836 seconds/update;
see [the findings](../findings/b200_mlp_gemm.md).

The normal Trainer installs this recipe through
`student.training.native_fp4_mlp=true` and
`student.training.native_fp4_mlp_parity_policy=selected_finite`. It no longer
needs the experiment's monkeypatch entrypoint or timing-only environment flags.
Compatible runs may use fixed step counts or ordinary epochs; experimental
timing-only kernels retain their separate twenty-step limit. Reuse
`results/b200_mlp_gemm/warmed03/causal_adapter/training_metadata.json`, SHA256
`14ab15279bb8895cf32353117d5c1cf957ad45d2b0ca9d7205067db27d77edeb`.
Check the receipt, kernel source hashes, B200/FA4 runtime and pinned cuDNN GEMM
source/runtime without repeating model numerical probes. Changed precision,
model revision, adapter layout, packing, compilation or context envelope fails
closed. A BF16 receipt cannot authorize native FP4, or vice versa.

Preserve raw failed strict packing/gradient and loss-parity receipts, prior
finite timing acceptance and passing isolation/memory preflight. Record the
new selection as `reuse_selected_finite_native_fp4_recipe`, with loss/gradient
parity explicitly waived; it is not a new parity pass. Finite/missing-gradient
checks remain active on every update. Short-run losses and warmed speed do not
establish long-run stability or monitoring quality.

Launchers expose the retained cuDNN Frontend 1.31.0 overlay (upstream revision
`51d9d06b574222378a3d806009accab098e73705`), cuDNN 9.26, Torch 2.11.0/cu130,
Triton 3.7.1 and PEFT 0.19.1. Reuse `.cache/training/shared/gpu-0/` compiler
caches, `.cache/training/shared/{cudnn_frontend,cute_dsl}` and the existing FA4
cache on the network volume; metadata records actual runtime/cache paths.
Keep the optimization worker resident across compatible trials. This wiring
does not start a training campaign or replace frozen jobs' recorded recipe
and checksums. Existing campaigns that reference the old default can explicitly
select the named BF16 profile when reconstructing their original configuration.
