# Corrected reasoning-SFT data scaling

This experiment re-evaluates the eight varied-only reasoning-SFT adapters from
the earlier Aletheia's Quest project. The historical 1--100% curve is invalid:
the checkpoints used causal-LM tensor paths under `model.layers`, while its
shared vLLM evaluator instantiated Qwen3.5 through the multimodal tree under
`model.language_model.layers`. AQ's later fingerprint audit showed that the old
vectors were largely the unadapted base model.

## Hypothesis and intervention

The intervention is the number of privileged-summary SFT examples: 36, 54,
144, 288, 576, 1,152, 2,301, and 2,877 rows (1, 2, 5, 10, 20, 40, 80, and
100%). Backbone, rank-16/alpha-32 LoRA, one epoch, AdamW `5e-5`, effective batch
32, data ordering, targets, and seed 0 are immutable historical artifacts. This
is a corrected retrospective measurement, not a retraining campaign.

The primary metric is macro per-dataset AUROC on the same frozen 822-row,
21-unit internal competition validation set used by the Kimi logit scaling
curve. Macro balanced accuracy at a fixed 0.5 terminal-margin threshold,
calibration, ties, scenario metrics, generated-decision agreement, and parse
failures are diagnostics. A single seed is sufficient to test the earlier
flat-curve claim; it does not estimate training variance.

## Evaluation protocol

Each immutable adapter is copied non-destructively and rebased to the Qwen3.5
multimodal language-model path. Source and serving weight checksums, serving
config checksum, and all 256 canonical tensor names are verified before vLLM
starts. A matched base pass and per-adapter effect fingerprint fail closed on a
silently ignored LoRA.

For every row, the model greedily generates up to 512 tokens under its exact
historical reasoning-summary prompt. The evaluator removes only the terminal
generated `0` or `1`, retains the generated reasoning prefix, and performs one
constrained next-token pass for literal `0` and `1` at that position. AUROC is
computed from their normalized continuous margin. It never conditions a second
score on the already-selected decision token.

## Held-out rule and stop condition

No validation row is used for training, prompt selection, checkpoint selection,
threshold tuning, or subset construction. Do not inspect the old local test
split. Stop if a source checksum changes, a serving key is noncanonical, the
base/adapter fingerprint is identical, a terminal margin is missing, or any
adapter is absent. Report a rejected power-law fit rather than forcing an
extrapolation when the curve is non-monotonic or has no positive exponent.

## Reproduction

On the source system:

```bash
python experiments/reasoning_sft_scaling/prepare.py
python scripts/lambda_cloud.py push --campaign monitor-foundation \
  results/reasoning_sft_scaling
```

On the reserved Lambda instance, use one persistent vLLM engine:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  experiments/reasoning_sft_scaling/run_lambda.py --gpu 1
```

Ignored results are written under `results/reasoning_sft_scaling/`; runtime logs
belong under `logs/lambda/reasoning_sft_scaling/`.
