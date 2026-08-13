# Decision log

## 2026-08-13 — Advance two Muon rates to replicated confirmation

- The rank-64, 25%-data internal-development screen completed all ten paired
  cells. Muon `5e-5` won the predeclared primary metric with 0.94727 macro
  AUROC, +0.00312 over AdamW at the same rate. Muon `1e-4` was the strongest
  multi-metric alternative, with 0.94608 macro AUROC, 0.88942 macro balanced
  accuracy, and 0.08900 macro Brier.
- Retain both Muon `5e-5` and `1e-4` for replicated 50%-data confirmation. Keep
  AdamW `5e-5` as the control. Do not promote Muon or evaluate a selected recipe
  on competition validation from this single-seed, non-lineage-grouped screen.
- Treat the reversal across rates as evidence that RMS matching does not remove
  optimizer-specific learning-rate sensitivity: Muon was worse at `1e-5` and
  `2e-5`, and better in macro AUROC from `5e-5` through `2e-4`.

## 2026-08-13 — Compare Muon through scheduled update-RMS matching

- Replace the independent, unscheduled Muon learning rate with the standard
  scheduler-managed parameter-group `lr`. Apply the per-matrix Moonshot scaling
  `0.2 * sqrt(max(A, B))`, so AdamW and Muon can be compared at the same five
  base learning rates despite LoRA A/B shape differences.
- Screen paired AdamW and Muon rates on rank 64, 25% training data, and a fixed
  internal dataset-label-stratified development split. Do not use competition
  validation for hyperparameter selection. The source artifact has no available
  conversation/generator lineage key, so record that the internal split cannot
  enforce lineage grouping.
- Keep the linear schedule, 3% warmup, two epochs, QLoRA/FLA path, batch, seed,
  data, and objective fixed. Treat the ten-cell screen as pruning evidence and
  require replicated confirmation before promotion.

## 2026-08-12 — Measure data and adapter capacity jointly

- Cross nested `10%`, `25%`, and `50%` training selections with QLoRA ranks 4,
  16, 64, and 256 using paired seeds 0, 1, and 2. Reuse the matching 100%-data
  cells from the completed capacity sweep, yielding 36 new training jobs and 12
  checksum-preserved references.
- Hold the proven NF4/double-quantized/BF16, FLA 0.5.2, microbatch-8 recipe fixed
  so the only interventions are data volume and adapter rank. Keep two epochs,
  meaning optimizer updates scale with data volume rather than being held fixed.
- Treat macro AUROC and within-seed rank contrasts as primary evidence. Estimate
  whether the paired rank-256 advantage over rank 16 grows with log training
  rows. Do not select a deployment rank on this validation surface.

## 2026-08-11 — Require FLA and standard QLoRA for the rank curve

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

## 2026-08-11 — Train text-only LoRAs causally and export serving copies

- For experiments containing no image or video inputs, train Qwen3.5 LoRAs
  through its causal text compatibility model and preserve that FP32 adapter as
  the master artifact.
- Export a checksum-tracked serving copy by deterministically rebasing causal
  PEFT keys into the multimodal wrapper's `language_model` namespace. Require a
  serving smoke test before promotion. This policy does not apply to multimodal
  training data or full-model fine-tuning.
- Retain full-sequence logits for the restarted curve. A matched 20-step H100
  preflight found that `logits_to_keep` was 2.9% slower (309.3 versus 300.7
  seconds) and produced a slightly different accumulated loss. Keep the
  selected-position mode only as an explicit benchmark condition.
- The native-multimodal adapter-capacity sweep was cancelled before evaluation
  and superseded by a fresh causal-master sweep; do not combine its checkpoints
  with the restarted scaling curve.

## 2026-08-10 — Match multimodal training and serving architectures

- Load multimodal backbones such as Qwen3.5 through
  `AutoModelForImageTextToText` during training, even for text-only batches, and
  restrict LoRA targets explicitly to the language model.
- Require an inference parity canary that demonstrates a meaningful adapter
  effect before evaluating a new model/adapter/backend combination at scale.
- Treat the first Kimi data-scaling validation pass as invalid: its adapters were
  trained against Transformers' causal compatibility wrapper, while vLLM served
  the multimodal conditional-generation wrapper. Rebase those adapter keys with
  source and destination checksums rather than modifying the master adapters.

## 2026-08-09 — Bootstrap

- Start Gleipnir as a standalone repository rather than retaining competition
  submission and leaderboard machinery.
- Use Qwen 3.5 as the initial backbone family.
- Seed the project with deception distillation, generic OpenRouter binary-logprob
  collection, Slurm tooling, and Lambda Cloud campaign management.
- Pin vLLM 0.24.0 and Transformers 5.14.1, the latest stable PyPI releases checked
  at bootstrap; exclude NNsight and NDIF.
- Keep experiments isolated and record durable findings under `docs/`.
