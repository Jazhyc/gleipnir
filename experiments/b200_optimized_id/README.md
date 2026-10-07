# Optimized-stack ID score drift

User-requested fixed-checkpoint comparison, 2026-10-08. Evaluate the latest
completed full-epoch 4B FP4/FA4 monitor adapter (272 updates, causal master
`8dbc1a2e...`, serving adapter `d13be8b2...`) using the optimized vLLM default.
Later twenty-update FROST timing adapters are scratch controls, not selected
trained monitors. Reuse the existing exact-source merged BF16 checkpoint.

Hypothesis: the accepted FP4/MXFP8/merged/two-row scoring stack changes scores
relative to this same adapter's cached BF16 dynamic-LoRA ID predictions. Use
all 3,012 canonical CoT-removed ID examples (946 STRIDE, 2,066 Gloom), the frozen
regular instruction, original order, labels, source lineage, non-thinking chat
boundary and `Prediction:` suffix. Require every source/rendered prompt hash and
token count to match the archived predictions before starting the server.
Selection is the already frozen final one-epoch checkpoint. No training, ID
tuning, threshold fitting, checkpoint/backend selection, OOD or promotion.

Compare against the single archived same-adapter BF16 pass, not another trained
model and not the training-seen full320 optimization controls. Preserve original
128-row batch partitions and issue at most 128 concurrent localhost score
requests within each partition. Run one full pass, as requested. Exclude a
training-source quick64 warmup; label the ID pass first-use-inclusive.
Repeat variation is unmeasured.
Persist each completed batch and preserve all per-request margins/scores/times;
failed partial passes remain diagnostic and are not mixed into complete reruns.
Report the
pass's input tokens/s, requests/s and latency; checkpoint writes remain included
in pass wall time and individual request latency excludes semaphore waiting.

Keep native Gigatoken/direct FROST, repaired Triton metadata, FP4 projections/
MLPs/direct SwiGLU, MXFP8 attention, BF16 operands/cache, FP32 gates/state,
causal LAST two-row head, prefix caching off, 32K context/chunk budget and
128 engine sequences. Apply the source-bound pooling token-reservation fix.
The old BF16 evaluation used dynamic LoRA, prefix caching on, 16 engine sequences
and 15% GPU utilization on another B200 host; this is whole-stack drift, not an
isolated quantization or matched speed attribution.

Bind canonical input/manifest, latest completed training summary, master/rebase,
old predictions/metrics and tokenizer/merged-source receipts. Reuse unchanged
native validations and persistent caches. Recheck the accepted real 20-row
adapter canary (MAE <=0.005, correlation >=0.995, nonzero adapter effect), and
report its separate frozen BF16-master comparison, preserving strict failures.
Stop on identity/truncation/missing/nonfinite/native/canary failure or completion;
a finite held-out score difference is the measurement, not a tuning signal.
Collect results and retire only this evaluation server afterward, leaving the
Pod, original checkpoints, shared caches and previous receipts intact.

```bash
PYTHONPATH=src:. python -m experiments.b200_optimized_id.prepare
# Upload its frozen binding, ID inputs/references and this directory only.
PYTHONPATH=<frozen-serving-source-bootstrap>:src:. <serving-python> \
  -m experiments.b200_optimized_id.run --name id01
PYTHONPATH=src:. python -m experiments.b200_optimized_id.analyze results/b200_optimized_id/id01
PYTHONPATH=src:. python -m experiments.b200_optimized_id.plot results/b200_optimized_id/id01
```

Results include pooled/source-macro/per-source raw normalized pAUROC@20 and
AUROC, Brier/ECE/log loss, fixed-0.5 threshold diagnostics, score/margin drift
percentiles/correlations, ties and fixed-threshold flips. Do not overwrite
frozen BF16 controls or the selected systems baseline.

If repetition count changes during startup, stop only the preparation driver,
retain the independently loading server and mark that attempt superseded.
Preserve any partial batches as diagnostics and run one complete pass.
`--reuse-prepared <source-run>` permits only a repetition-count change, verifies
the prepared input and live API identities, and reuses the server.
The original source/settings receipts remain intact. Batch diagnostics report
checkpoint-write time and request completion percentiles; these identify barrier
tails but do not isolate their causal throughput cost.
