# ID-selected three-epoch MIL: frozen OOD evaluation

Status: all 6,395 frozen OOD rows evaluated; completion audited on 2026-09-08.

The selected MIL adapter reached source-macro pAUROC@20 `0.781299`, versus
released 4B's `0.782350` (difference `-0.001051`); macro AUROC was `0.906762`
versus `0.908584`. This does not establish an OOD improvement. The comparison
changes training mixture, duration and objective together and is not an isolated
MIL effect. Complete predictions, source breakdowns, calibration, threshold
diagnostics, score ties and merge provenance are retained under
`results/monitoring_mil_ood/ood_evaluation/4b/adapters/mil-pct020-lr2em05-epochs3-seed0/`.
The master and evaluation evidence are covered by the
[campaign inventory](../../docs/findings/gleipnir_improvement_shutdown_inventory.md).

Hypothesis: the ID-selected monitoring-only MIL checkpoint transfers better
than released Gleipnir 4B. Evaluate exactly
`mil-pct020-lr2em05-epochs3-seed0`: 1,738 monitoring parents, three epochs,
LR 2e-5, logmeanexp MIL weight .25. The mixed-deception variant is excluded.
The selection is fixed before this run. No training, prompt/threshold changes,
additional candidates, or OOD-driven hyperparameter selection are authorized.

Reuse all 6,395 frozen six-source OOD rows, compact student rubric, pinned
Qwen3.5-4B revision, and direct one-token 0/1 logprobs. Primary metric is
source-macro raw normalized pAUROC@20; report AUROC, Brier, threshold diagnostics,
ties and source breakdowns. Compare descriptively against the historical released
4B compact-prompt result (.782350 pAUROC, .908584 AUROC). This comparison changes
data mixture, duration and objective together; it cannot isolate MIL causally.

The entrypoint freezes a single-job manifest and evaluation config, validates
training completion and master/export checksums, then runs the original balanced
12-row master-versus-vLLM parity gate (Pearson >=.99, MAE <=.02, nonzero adapter
effect). Only after passing, split whole 128-row batches across two independent
TP1 vLLM engines on the existing Lambda H100 SXM5 GPUs. Preserve FlashInfer GDN,
continuous batching, unchanged scoring and isolated resumable prediction files.
The existing bounded evaluator watchdog may recover owned stalled processes.
Stop on artifact drift, failed parity, missing/nonfinite scores, failed workers
or incomplete/duplicate merge. Recompute metrics only after exact full coverage.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.monitoring_mil_ood.run
```

Artifacts: `results/monitoring_mil_ood/`. Supervisor log:
`logs/lambda/monitoring_mil_ood/run.log`. This session has no timed agent-followup
tool; the execution supervisor/watchdog is not an agent heartbeat.

## Launch record — 2026-09-07

Supervisor PID 232256 on `gleipnir-improvement`. Frozen input checksum verified;
exact MIL3 master/export parity passed (Pearson .999704, MAE .003356,
maximum score difference .020887; nonzero adapter effect). Full sharded
evaluation began around 20:53 UTC. At startup verification both H100s were at
100% utilization with 128/256 saved rows on GPUs 0/1 respectively. Three focused
contract/sharding tests passed locally; the initial two-test suite also passed
on Lambda. No training was restarted and no OOD result was used for selection.
