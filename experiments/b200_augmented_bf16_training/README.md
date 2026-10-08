# Augmented training with BF16 MLPs and FA4

Hypothesis: native FP4 MLP forward/base input gradients account for part of the
new augmented adapter's ID regression. Train the exact same 8,688 augmented
examples (3,475 injected) from the original seed-0 zero-B rank-128/alpha-256 FP32
initializer, with unchanged Kimi soft targets and insertion ledger. Use one epoch,
272 logical updates of 32, AdamW 5e-5, linear decay/3% warmup, no dropout/weight
decay/checkpointing, and the same 16,384-token packing/context cap. Training must
cover all 83,969,213 tokens with finite, complete gradients.

Intervention: use `qwen35_4b_b200_bf16_fa4`, retaining BF16 FlashQLA, native varlen
FA4 4.0.0b33, Torch 2.11 and selective compilation. The MLP base forward and input
gradients use BF16. LoRA masters remain FP32. Compare resolved jobs before launch:
only MLP precision, associated validation policy/reference and output bookkeeping
may differ. Reuse the checksum-bound BF16/FA4 receipt and persistent compiler
caches; its accepted 10% gradient ceiling is diagnostic policy, while the FP4
receipt records a failed strict 5% comparison. Neither is a fresh parity pass.

Select the final complete one-epoch checkpoint only; no tuning or promotion on
ID. Preserve FP32 masters, merge in FP32 to a separate BF16 checkpoint on ephemeral
storage, and require master/merged/serving parity (MAE <=0.02, correlation >=0.99,
adapter effect >1e-6, finite scores). Serve with the explicitly requested BF16
exception: unquantized MLP/GDN/attention projections and actual BF16 FlashInfer
attention, exact causal LAST two-logit head, vLLM 0.31, original 128-row groups,
concurrency 128 and prefix cache off. Score all 3,012 canonical CoT-removed ID
examples, 33,750,959 tokens, once. No APPS scoring is part of this control.

Primary baseline: archived FP4-trained augmented checkpoint under this same
BF16 serving recipe/NC2 host, macro raw pAUROC@20 0.839096. Report paired changes,
source-macro/pooled/per-source AUROC and raw normalized pAUROC@20, calibration,
ties, fixed-0.5 diagnostics, latency and input tokens/s. Historical augmented and
regular scores are qualified context. One seed and prior ID exposure limit
causal/generalization claims; do not select a production recipe from this pass.

Stop on drift beyond the permitted intervention, nonfinite/missing gradients,
OOM, failed adapter/native precision checks, truncation, incomplete coverage or
completion. Preserve all failed receipts. Replace the current recorded scorer
sequentially after CPU preparation/runtime checks; keep successful serving warm.
Use the existing authorized NC2 B200 only and preserve all baselines/caches.
There is no in-chat heartbeat scheduler; monitoring occurs in the active turn.

Entrypoint: `python -m gleipnir.campaigns.monitoring prepare --config
experiments/b200_augmented_bf16_training/config.yaml`, then `run` with the same
configuration in the preserved runtime. No copied campaign controller is needed.
Outputs/data/logs use `b200-augmented-bf16-training02`; merged weights live at
`/tmp/gleipnir-merged/bf16-augmented-fa4`.
