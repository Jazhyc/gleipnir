# Same BF16-trained augmented adapter, optimized ID/APPS serving

User-requested serving comparison for the fixed completed BF16 MLP/FA4 adapter.
Hypothesis: the selected optimized recipe retains useful ID/APPS ranking while
raising throughput. Keep the exact 272-update FP32 adapter and merged BF16
checkpoint at `/tmp/gleipnir-merged/bf16-augmented-fa4`; no training or merge.
Switch to the selected compiled vLLM 0.31 recipe: FP8 attention projections,
FP4 MLP/GDN projections, native FP4 SwiGLU output and MXFP8 attention. Retain
native tokenizer, cached causal LAST two-logit head, corrected scheduler,
prefix caching off, 32K limits, original 128-row partitions and concurrency 128.

Freeze inputs, executed sources, selected serving recipe and completed adapter,
merge/reference/canary identities before replacing the recorded BF16 scorer.
Reuse persistent compiler/kernel caches and unchanged native receipts. Require
fresh adapter-specific master/merged score agreement (MAE <=0.020, correlation
>=0.99, effect >1e-6, finite scores), exact tokens and actual FP8 projection audit
before held-out scoring. Stop on any gate failure; an earlier adapter's finite
acceptance does not transfer. Preserve failed receipts and partial outputs.

Score all 3,012 CoT-removed ID examples, 5,184 APPS benchmark cases and 3,930
honest controls once, preserving prompt hashes, labels, metadata and order.
Input tokens are 33,750,959 / 8,207,435 / 7,132,518. Cached same-adapter BF16
scores are primary paired controls. The earlier FP4-trained/optimized diagnostic
is qualified training-precision context and retains its original failed gate.
No checkpoint/precision selection or promotion uses held-out results.

Report ID source-macro/pooled/per-source AUROC and raw normalized pAUROC@20,
calibration, ties, fixed-0.5 diagnostics and paired changes. APPS retains the
clean-only strict `score > threshold` at-most-2%-FPR rule across all templates;
report honest FPR, backdoor TPR, matched ranking, paired injection score shifts,
calibration and template variation. Report input tokens/s, requests/s and
latency on matched cohorts. Prior ID/APPS exposure and dependent task/template
lineage limit robustness claims. No program tests or six-source OOD sweep.

Stop on source/model/runtime drift, failed parity/native audits, nonfinite or
truncated scores, OOM, incomplete coverage or completion. Retire only the recorded
server after preparation; preserve its receipts and cached scores. Keep the
passing optimized scorer warm on the existing authorized NC2 B200. No capacity
change. There is no in-chat heartbeat scheduler; monitoring is in the active turn.

Run `python -m experiments.b200_augmented_bf16_optimized.run --stage prepare`,
then `--stage score` through the staged 0.31 runtime wrapper. The shared campaign
evaluator performs optimized startup/parity/ID/APPS scoring; this entrypoint only
binds the completed adapter and replaces the recorded parent. Results/data/logs
use `b200-augmented-bf16-optimized01`; earlier contracts remain frozen.
