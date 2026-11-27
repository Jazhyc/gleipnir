# Require FLA and standard QLoRA for the rank curve

Date: 2026-08-11.
Status: historical; governs no future work.

- The fast Qwen3.5 H100 path requires more than `AutoModelForCausalLM`: install
  isolated `flash-linear-attention==0.5.2` and `fla-core==0.5.2` packages before
  model import, verify Transformers selected their gated-delta kernels, and fail
  closed if the probe fails. Use the measured eager recipe of microbatch 8,
  accumulation 4, effective batch 32, gradient checkpointing, and no
  `torch.compile`.
- Train every adapter-rank condition with the standard QLoRA base configuration:
  4-bit NF4, double quantization, and BF16 compute. Hold quantization and batch
  fixed across ranks. Preserve the trained FP32 adapters and evaluate their
  checksum-rebased copies on the ordinary BF16 serving base.
- Use selected-position vocabulary projection for every QLoRA rank. A rank-256,
  batch-8 preflight consumed about 69 GB before full-sequence logits triggered a
  further roughly 25 GB allocation and OOMed. The earlier selected-position
  batch-2 path was 2.9% slower, but it is necessary to retain the proven batch
  and is held fixed across this new curve.
- Require a rank-256, 20-step memory and throughput preflight before scheduling
  the full curve. This catches FLA fallback and high-rank OOMs without silently
  changing the experimental recipe.
- The corrected single-GPU preflight measured 3.551 samples/s at rank 16 and
  3.524 at rank 256, with peak allocated memory of 26.19 and 33.25 GB. Treat the
  earlier 8.811 samples/s measurement as specific to its shorter 2,880-row
  prompt distribution; the full mixed-data runs project to about 2.1 hours.
- Retain seeds 0, 1, and 2 at every rank. In the prior full-data condition,
  macro AUROC had SD 0.00182 and range 0.00357, while the preceding 64%-to-100%
  mean gain was only 0.00256. Seed noise is therefore large enough to imitate
  the capacity effects this curve is designed to measure.
- Cancel and preserve the causal BF16/microbatch-2 attempt after its first two
  rank-256 completions. It used the slow Torch gated-delta fallback and is not
  combinable with the QLoRA curve.
