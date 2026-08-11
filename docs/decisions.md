# Decision log

## 2026-08-11 — Train text-only LoRAs causally and export serving copies

- For experiments containing no image or video inputs, train Qwen3.5 LoRAs
  through its causal text compatibility model and preserve that FP32 adapter as
  the master artifact.
- Export a checksum-tracked serving copy by deterministically rebasing causal
  PEFT keys into the multimodal wrapper's `language_model` namespace. Require a
  serving smoke test before promotion. This policy does not apply to multimodal
  training data or full-model fine-tuning.
- For direct final-token objectives, use `logits_to_keep` to project only the
  distinct supervised sequence positions in each batch. Retain a full-logit
  mode for numerical and throughput parity benchmarks.
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
