# Select native FP4 MLPs as the B200 training default

Date: 2026-10-05.
Status: superseded by [BF16/SDPA B200 training default](b200_bf16_sdpa_training_default.md).

- `qwen35_4b_b200_default` now selects the warmed native NVFP4 MLP recipe with
  BF16 FlashQLA/FA4 and FP32 master adapters. Ordinary training reuses its bound
  startup receipt with explicit finite acceptance, preserving failed strict
  loss/gradient checks. Retain the named BF16 comparison and shared caches.
  See the [recipe decision](b200_native_fp4_training_recipe.md).
