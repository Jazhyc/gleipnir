# Historical augmented adapter, optimized versus BF16 ID serving

User requested a matched optimized-stack ID evaluation of the historical
augmented 4B adapter that retains AUROC/pAUROC 0.962251/0.872693 on current
BF16 serving. Hypothesis: the selected optimized recipe changes its held-out
ranking and scores while improving throughput. Keep the exact historical FP32
master/export and `/tmp/gleipnir-merged/historical-augmented-bf16` checkpoint.
No training, new merge, prompt change, selection or promotion.

Reuse the completed fresh master/merged references and 20 training-source
canaries from `b200-historical-augmented-bf16-id01`. Require new optimized
master/merged parity (MAE <=0.020, correlation >=0.99, effect >1e-6, finite scores)
and actual native audits before the held-out pass. No prior adapter's diagnostic
authorization transfers. Stop on a failed gate, drift, nonfinite/truncated or
missing outputs, OOM or completion; retain failures and partial outputs.

Switch compiled vLLM 0.31 BF16 serving to the selected FP8 attention projections,
FP4 MLP/GDN projections, native packed FP4 SwiGLU output and MXFP8 attention.
Keep native tokenizer, causal LAST cached 0/1 head, FP32 gates/state, corrected
synchronous scheduler, CUDA GDN/automatic context parallelism, prefix caching
off, 32K limits, original 128-row partitions and c128. Score all 3,012 fixed
CoT-removed ID prompts and 33,750,959 input tokens once. The combined recipe
comparison does not isolate an individual precision or kernel component.

Primary paired control: the same historical adapter/checkpoint, same NC2 host
and matched BF16 ID workload. Recent BF16- and FP4-trained optimized replicas
and the historical older-stack scores are qualified secondary controls; the
FP4-trained diagnostic retains its original failed gate. Regular optimized ID
on the EU host is context. Preserve the metadata-only historical baseline view
and its original hashes/scores. Report macro/pooled/source AUROC and raw pAUROC20,
calibration, ties, fixed-0.5 diagnostics, paired score/decision shifts, input
tokens/s, requests/s and latency. Timing is one matched descriptive pass.

Prepare and validate all bindings before retiring only the recorded BF16 scorer.
Reuse persistent compiler/kernel caches and unchanged native receipts. Keep the
passing optimized 0/1 scorer warm on the existing authorized NC2 B200. No
capacity change or APPS/JudgeDeceiver scoring. Active-turn monitoring only;
there is no heartbeat scheduling tool.

Use the shared completed-adapter follow-up entrypoint:
`python -m experiments.b200_augmented_bf16_optimized.run --config
experiments/b200_historical_augmented_optimized/config.yaml --stage prepare`,
then `--stage score` in the selected 0.31 runtime wrapper. Artifacts use
`b200-historical-augmented-optimized-id01` under data/results/logs. Historical
BF16 controls and adapter files remain intact.

After the original canary failed at MAE 0.051628/correlation 0.973095, the user
explicitly authorized both violations for ID-only diagnostic scoring. Keep the
original contract/receipt intact; use `diagnostic_config.yaml` through the same
prepare/score entrypoint, writing separate
`b200-historical-augmented-optimized-diagnostic01` artifacts. Pin the original
failed gate and preserve strict limits in all new receipts. The separate
`allow_correlation_drift` exception applies only with diagnostic scope; finite
scores/correlation, adapter effect, native audits and exact identities remain
mandatory. This permits measurement, not checkpoint or recipe promotion.
