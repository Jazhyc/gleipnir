# Select uniform FlashQLA for B200 NF4/BF16 LoRA

Date: 2026-10-02.
Status: superseded by [BF16/SDPA B200 training default](b200_bf16_sdpa_training_default.md).

- Use `systems_screen@_global_: qwen35_4b_b200_default`, with all 24 GDN layers
  on FlashQLA, SDPA full attention and the selected checkpoint/adaptive batching
  recipe. The user selected the measured 12.99% lower step time and close
  ten-step losses. Retain original-FLA profiles and failed strict parity records.
- Ordinary training records finite acceptance explicitly and rejects missing or
  nonfinite gradients before updates. This decision does not establish held-out
  quality equivalence or long-run stability. See the
  [recipe decision](b200_flashqla_training_recipe.md).
