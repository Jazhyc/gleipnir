# Qwen3.8-27B B200 training default

User selection, 2026-10-17: train Qwen3.8-27B on one B200 with
`qwen38_27b_b200_default`, which aliases
`qwen38_27b_b200_packed_sdpa_trim_attn12`. No further training-speed
optimization is planned; precision stays BF16.

**Recipe.** The 4B default ([decision](b200_bf16_sdpa_training_default.md)),
with only what the 27B needs changed:
- BF16 MLPs, BF16 FlashQLA GDN with FP32 gates/norm (48 layers), segmented
  causal SDPA, rank 128/alpha 256 LoRA with FP32 masters, 16,384-token
  packing, logical batch 32;
- compiled final-layer trim (layer 63 computed only at decision positions),
  guarded by its pre-update canary;
- non-reentrant gradient checkpointing of all 48 GDN layers and attention
  layers 3, 19, 35 and 51; the other 12 attention layers keep their
  activations (the 27B needs ~330 GB uncheckpointed for a 16K pack);
- `checkpointed_full_attention_and_linear_shell` compilation, compile-cache
  limit 128.

**Evidence** ([experiment](../../experiments/qwen38_27b_training_speed/README.md)):
the ten-update packed benchmark, the same GPU, the same batches.
- **Speed.** 24.99 s per update against 27.80 s with every layer checkpointed
  (−10.1%; every batch −9 to −12%). The trim gives −3.1% alone and the 12
  stored attention layers −7.7%. One strict-filtered epoch is about 1.7 h of
  updates.
- **Correctness.** The trim canary passed: gradients within the repeat noise
  floor plus 1 point, logits within one BF16 ulp. Losses track the control
  within run-to-run noise. Storing layers does not change arithmetic.
- **Memory.** 151.8 GiB on the 28,733-token preflight and 154.6 GiB in
  updates, of 178 GiB. Storing all 16 attention layers ran out of memory.

**Qualifications.**
- Full-campaign rows reach 29,696 tokens, 3% beyond the benchmark's longest,
  which leaves about 20 GiB of margin.
- The FlashQLA model canary reads 0.34 gradient relative L2 against pinned
  FLA (4B: 0.13), concentrated in early layers; both are accepted as finite.
  Early-layer 27B LoRA gradients are sensitive to GDN arithmetic.
- **Not yet validated for campaigns.** The first 27B campaign must run full
  startup validation (no 27B startup receipt exists to reuse), freeze a 27B
  initial-adapter digest, and validate merge and scoring at 27B.
