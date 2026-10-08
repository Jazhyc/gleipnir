# Same augmented checkpoint, BF16 ID control

Hypothesis: low-precision serving explains part of the augmented checkpoint's
ID regression. Score the same completed FP4-trained augmented adapter's merged
BF16 weights with unquantized BF16 MLP/GDN/attention projections and BF16
FlashInfer attention. Preserve FP32 recurrence state/gates, CUDA GDN/automatic
context parallelism, native tokenizer, causal LAST two-logit head, corrected
vLLM 0.31 stock scheduler, prefix caching off, compiled execution and 32K limits.

Reuse the original ordered 3,012-row CoT-removed ID workload, 128-row partitions
and concurrency 128. Compare paired predictions with the same-host optimized
augmented diagnostic; cached regular optimized ID and historical augmented BF16
are qualified secondary controls. The checkpoint, tokenizer assets, prompt hashes,
labels and 33,750,959 input tokens stay fixed. No training, APPS, checkpoint or
precision selection/promotion is part of this request.

Before held-out scoring, require all projection weights to be BF16/unquantized,
64 MLP/48 GDN/16 attention projection coverage, exact classifier source rows,
eight actual BF16 query/cache attention calls and the unchanged adapter-specific
master/merged parity limits (MAE <=0.02, correlation >=0.99, effect >1e-6).
Stop on identity/source/runtime drift, failed audits/parity, nonfinite output,
truncation, missing membership, OOM or completion. Preserve failed attempts.

The user explicitly requests this BF16 serving exception to isolate the current
optimized-stack effect. It is BF16 inference of the existing FP4-trained model;
it does not test BF16 training. Replace the current scorer sequentially after
preparation and retain the passing BF16 control warm on the existing B200.
Capacity and persistent caches are unchanged.

Run `python -m experiments.b200_augmented_bf16_id.run --stage prepare`, then
use the staged 0.31 runtime wrapper with the same module and `--stage score`.
The shared campaign evaluator runs serving/parity/ID scoring; the entrypoint
only binds the already completed adapter and cached comparison. Results/data
live under `results/b200-augmented-bf16-id01/` and
`data/b200-augmented-bf16-id01/`; logs under
`logs/runpod/b200-augmented-bf16-id01/`.
