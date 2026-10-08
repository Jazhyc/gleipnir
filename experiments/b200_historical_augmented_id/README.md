# Historical augmented adapter on current BF16 ID serving

User requested evaluation of the preserved historical augmented Qwen3.5-4B
adapter on the current ID set using BF16 serving. Hypothesis: the historical
adapter retains its ID advantage when served through the current vLLM stack.
Keep its fixed final 272-update FP32 master and rebased FP32 export; no training,
checkpoint selection, prompt changes, precision sweep or promotion.

The source is `results/monitor_injection_augmentation/4b/augmented`, with master
SHA256 `3ecfd0d397aed7ad666e189bbd6fed82a20f93be33b5608a32c288c78ac7edef`
and export `bbed5d2c6fc5c9e8ae84a78993d43b2151acfd306e1d14b7ce5013e8e3c17fd7`.
Its historical BF16/SDPA training metadata is authoritative. The campaign's
BF16/FA4 profile supplies input checks and the shared reference environment;
it is not a description of new training. A new output wrapper links the frozen
adapter directories and preserves its completion receipt; references are written
in that wrapper, leaving historical files intact.

Freeze the canonical 3,012 CoT-removed ID prompts, labels, lineage, order,
rendered hashes and 33,750,959 input tokens. Match the recent BF16 controls:
compiled vLLM 0.31, native tokenizer, cached causal LAST 0/1 head, BF16 MLP/GDN/
attention projections and attention query/cache, FP32 state/gates, CUDA GDN,
automatic context parallelism, corrected synchronous FCFS scheduler, prefix
caching off, 32K envelope, 128-row partitions and c128. This explicit user-requested
BF16 control is separate from the selected optimized serving default.

Reuse the shared campaign stages: bounded FP32-master reference on deterministic
training-derived canaries, FP32-accumulated BF16 merge on ephemeral storage,
merged/master parity, then native-audited BF16 serving and complete ID scoring.
Require finite scores, MAE <=0.020, correlation >=0.99, adapter effect >1e-6,
all BF16 projections, exact head rows and eight actual BF16 attention calls.
Stop on failed parity/audits, drift, missing/truncated/nonfinite rows, OOM or
completion. No previous diagnostic authorization transfers to this adapter.

Primary controls are the recent FP4-trained/BF16-served and BF16-MLP/FA4-trained/
BF16-served augmented adapters on NC2. Reuse the historical adapter's old ID
scores as the paired stack comparison. The regular optimized EU ID reference
is qualified context. Report macro/pooled/source AUROC and raw normalized
pAUROC@20, calibration, ties, fixed-0.5 diagnostics, paired score/decision changes,
input tokens/s, requests/s and latency. One pass is descriptive timing.
Historical cached scores use a different provenance nesting in 946 rows. Its
derived baseline view rewraps metadata only after exact ID, label, source,
rendered-prompt hash and token checks; original metadata and scores are preserved.

Use the existing authorized NC2 B200. Retire only the recorded A/B scorer after
preparation, preserving its results, checkpoint and caches. Archive the unused
FP4-augmented merged checkpoint on the persistent volume with exact checksums
to make ephemeral room; preserve masters and merge receipts. Keep the passing
historical BF16 0/1 scorer warm. No capacity change. No heartbeat scheduler is
available; monitoring occurs in the active turn.

Run `python -m experiments.b200_historical_augmented_id.run --stage prepare`
remotely, then `--stage all` in the staged 0.31 runtime wrapper. Artifacts use
`b200-historical-augmented-bf16-id01` under data/results/logs. The new merged
checkpoint is `/tmp/gleipnir-merged/historical-augmented-bf16`.
