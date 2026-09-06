# Monitoring-only training duration

## Frozen question (2026-09-05)

Does a fresh two-epoch soft-distillation schedule improve on the selected
one-epoch recipe? Train exactly two seed-0 cells, initial LR 1e-5 and 2e-5,
in parallel on the two reserved Lambda H100 80GB GPUs (driver 580.126.20,
both idle at pre-launch probe). No MIL, rationale CE, hard-label auxiliary,
deception data, or new teacher calls. The auxiliary-objective line is closed
for now at the user's request.

Reuse the monitoring LR screen's selected v6 recipe: 8,688 cached monitoring
rows, Qwen3.5-4B pinned revision, rank128/alpha256 NF4 double-quantized QLoRA,
BF16 compute, standard AdamW, microbatch1/accumulation32, linear-attention-only
checkpointing and full_attention_and_linear_shell compilation. Keep pinned
FLA0.5.2, causal-conv1d1.6.2.post1, Triton3.7.1 and disabled FLA dispatch.
This is the matched faster soft-only recipe, not the rationale memory recipe.

Only initial LR and duration differ from the historical cells. Start from the
base, not from a one-epoch adapter; use a fresh two-epoch linear decay with 3%
warmup, zero weight decay and dropout. There are 544 optimizer steps, including
each epoch's partial final batch. Evaluate final checkpoints only. Do not select
intermediate ID peaks or extend training based on observed ID results.

## Comparison and stopping

Compare both final adapters against the historical one-epoch 2e-5 default
(macro pAUROC@20 0.8710662298), and report the one-epoch 1e-5 result as duration
context (0.865921). Reuse the frozen 3,012-row ID suite and compact decision
prompt. A promising candidate must gain at least 0.005 macro pAUROC@20, lose
at most 0.01 on either source, and regress macro Brier by at most 0.005.
These are exploratory development criteria, not automatic release promotion.
No strict OOD evaluation, extra LR cells, or auxiliary combinations are queued.

Fail closed on provenance drift, unavailable pinned kernels, longest-32-row
preflight failure, compile-canary failure, training metadata drift/nonfinite
loss, serving-parity failure, or incomplete/nonfinite ID predictions. Retain
FP32 master adapters separately from serving exports. One persistent vLLM
engine evaluates both completed adapters after a bounded master/export parity
gate. Poll startup promptly, then use 10-minute monitoring heartbeats.

## Implementation

One entrypoint reuses LR-screen preparation/recipe validation, shared status,
training/rebasing, and the existing parity/evaluation pipeline. Hydra YAML is
resolved once into a checksummed JSON contract; runtime does not reread YAML.
No historical manifests or adapters are overwritten.

The Python `execute` function also accepts explicit job-factory, completion-
validator, and log-root hooks for the prefix-supervision campaign. Its default
duration behavior and historical artifact contracts are unchanged.

```bash
python -m experiments.monitoring_duration.run prepare
python -m experiments.monitoring_duration.run run --revision COMMIT
```

Artifacts: `results/monitoring_duration/`; logs:
`logs/lambda/monitoring_duration/`. Each training lane has its own log.

## Completed result (2026-09-06)

Both final 544-step checkpoints completed the full 3,012-row ID evaluation.

| Recipe | Macro pAUROC@20 | Macro AUROC | Macro Brier |
| --- | ---: | ---: | ---: |
| Historical 2e-5, one epoch | 0.871066 | 0.957643 | 0.079273 |
| 1e-5, two epochs | 0.854856 | 0.956150 | 0.081885 |
| 2e-5, two epochs | 0.882952 | 0.963164 | 0.073997 |

The 2e-5 two-epoch endpoint passes the predeclared exploratory gate: macro
pAUROC gain +0.011886, Gloom +0.021094, STRIDE +0.002677, and improved macro
Brier. The 1e-5 endpoint worsens both source rankings. Thus this screen does
not support a blanket conclusion that longer training overfits: its effect
depends on the learning-rate/schedule configuration. This is one seed on the
development set, not a demonstrated OOD gain or automatic recipe promotion.

FlashInfer evaluation stalled twice, at 2,816 saved rows for 1e-5 and 384 for
2e-5. Only evaluator processes were restarted; all training checkpoints,
frozen configurations, and completed prediction shards were preserved. The
backend was unchanged. The remote heartbeat initially logged unchanged state
without detecting stalls; manual checks recovered both incidents. Resumed
batch scheduling may cause small numerical differences, so retained outputs
and recovery records are authoritative. The final aggregate and complete
prediction artifacts have been collected locally under the campaign root.
