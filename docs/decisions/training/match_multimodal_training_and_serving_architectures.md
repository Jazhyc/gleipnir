# Match multimodal training and serving architectures

Date: 2026-08-10. Status: recorded in the decision log.

- Load multimodal backbones such as Qwen3.5 through
  `AutoModelForImageTextToText` during training, even for text-only batches, and
  restrict LoRA targets explicitly to the language model.
- Require an inference parity canary that demonstrates a meaningful adapter
  effect before evaluating a new model/adapter/backend combination at scale.
- Treat the first Kimi data-scaling validation pass as invalid: its adapters were
  trained against Transformers' causal compatibility wrapper, while vLLM served
  the multimodal conditional-generation wrapper. Rebase those adapter keys with
  source and destination checksums rather than modifying the master adapters.
