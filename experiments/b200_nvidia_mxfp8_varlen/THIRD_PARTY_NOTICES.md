# NVIDIA source attribution

The project-owned variable-length quantizer and pointer host adapt NVIDIA
cuDNN Frontend revision `51d9d06b574222378a3d806009accab098e73705`:

- `python/cudnn/gated_attention_block/kernels/quantize_mxfp8.py`, Apache-2.0.
- `python/cudnn/sdpa/bwd/kernels/sm100/prepared_mxfp8_host.py`, Apache-2.0.
- Scale-factor addressing follows
  `python/cudnn/sdpa/bwd/kernels/sm100/bprop_sf_repack_mxfp8.py`, MIT.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
The applicable license texts accompany this notice as
`LICENSE-NVIDIA-Apache-2.0.txt` and `LICENSE-NVIDIA-MIT.txt`.

Gleipnir changes: sequence-local cumulative offsets, packed forward scale tiles,
runtime-shaped backward views, whole-row scale conversion and exact singleton
semantics. The native forward/backward kernels remain dependencies in the
isolated upstream overlay; no kernel-body relicensing is asserted.
