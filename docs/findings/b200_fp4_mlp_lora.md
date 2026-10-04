# Native FP4 MLPs after leaving QLoRA

Date: 2026-10-04. Systems screen in progress on the existing US-NC-2 B200.
Native arithmetic and per-token scaling checks pass. Whole-model packing,
compilation, finite updates and throughput remain unestablished.

## Intervention and control

Load Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
without bitsandbytes or k-bit preparation. Replace only the 96 frozen MLP
gate/up/down bases with Four Over Six 1.0.5 native CUTLASS W4A4 forwards.
Use strict per-token 4/6 activation selection and fused scaling. Input gradients
use the exact decoded forward weights in BF16. The other base components retain
BF16, and trainable rank-128/alpha-256 LoRA master parameters retain FP32.
Original BF16 MLP weights and decoded backward weights remain resident;
this does not establish a four-bit memory footprint or FP4 backward.

The control is the completed clean segmented-SDPA condition in
`results/b200_bf16_fa4_accepted/summary.json`: final ten updates average
5.123633625 seconds, with peak allocated memory 146.995065689 GiB. Preserve
the frozen 320-row selection, soft targets, initial master tensor digest
`a6b1d2e9fd89efff9523150a76035a2e5d27900eaae3c7a4820e3b9277078f11`,
logical batch 32, 16,384-token packing, 29,696 context, 24 FlashQLA GDN layers,
selected-token projection, optimizer and scheduler. No model checkpointing.
Require exact physical batch/token agreement before comparing measured updates.

The [experiment contract](../../experiments/b200_fp4_mlp_lora/README.md)
defines 20 updates, ten warmup, and a 5% improvement threshold for recommending
follow-up. Strict packing gradient tolerance stays 5%; the earlier explicit
FA4 learning acceptance does not authorize relaxing this FP4 gate. No teacher
calls or held-out evaluations occur in this systems screen.

Runtime retains the existing B200, Torch 2.11.0+cu130, Transformers 5.14.1,
PEFT 0.19.1, FLA/fla-core 0.5.2, FlashQLA revision
`da06429d54b0f577de0a638f451ac8f0b395e0ac` and Triton 3.7.1.
Four Over Six uses its isolated 1.0.5 native extension. Kernel and compiler
caches reuse `.cache/training/shared/gpu-0` on the persistent network volume.
Commands, source archives, input checksums, effective cache paths and receipts
are recorded with each attempt.

## Startup evidence

Fused normalization/rescaling is bitwise identical to the unfused row reference
on widths 2560 and 9216, irregular row counts and the 16,384-token shape.
Native batch-1/2/4/8 forward relative L2 against decoded reference arithmetic
is 0.002301–0.002345; decoded-BF16 backward error is 0.001647–0.001664.
Row independence error is exactly zero. These validate local arithmetic,
not agreement of the FP4 model with the original BF16 model.

The first Trainer attempt stopped before any update: unquantized loading kept
the model on CPU until Trainer placement, whereas native weight conversion
requires CUDA immediately. It is preserved under `results/b200_fp4_mlp_lora/`.
The repair moves the original BF16 base and FP32 adapters to CUDA before native
conversion. Its separate outputs are `results/b200_fp4_mlp_lora_cuda/` and
`logs/runpod/b200_fp4_mlp_lora_cuda/`.

The focused CPU suite passed 87 tests on the pod with CUDA hidden, including
mixed-precision scope, loading rejection, packing contracts, frozen workload
construction and existing BF16/FA4 behavior. Local tests were obstructed by
filesystem I/O waits during Torch import; the remote test receipt is retained.
