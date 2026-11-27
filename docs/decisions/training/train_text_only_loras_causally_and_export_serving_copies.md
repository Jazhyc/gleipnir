# Train text-only LoRAs causally and export serving copies

Date: 2026-08-11. Status: recorded in the decision log.

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
