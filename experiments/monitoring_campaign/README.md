# Monitoring campaign runner

Use `python -m gleipnir.campaigns.monitoring` for fresh, one-epoch
Qwen3.5-4B campaigns on an existing authorized B200. The runner uses the selected
native FP4 training profile, preserves FP32 adapters, merges a BF16 checkpoint
on ephemeral storage, checks that adapter's master/merged/optimized agreement,
and runs registered `id` and/or `apps` evaluations. It does not provision or
terminate capacity. GPU stages run in separate training and serving processes.

Copy [config.yaml](config.yaml) and change the campaign name, experiment fields,
input paths/checksums, initializer, GPU identity and runtime paths. The example
replays the original 40% injection augmentation. Its hypothesis is that this
reduces benign APPS injection false positives while preserving ID ranking; the
baselines and their host/precision qualifications are explicit in configuration.
Selection is the final one-epoch adapter, with no held-out selection. Stop on
input/source/runtime drift, nonfinite training, missing coverage or failed parity.
The YAML is a launchable example, not a receipt that it has been trained.

```bash
python -m gleipnir.campaigns.monitoring inventory --config path/to/campaign.yaml
python -m gleipnir.campaigns.monitoring prepare --config path/to/campaign.yaml
python -m gleipnir.campaigns.monitoring run --config path/to/campaign.yaml
```

`inventory` reports all missing or mismatched declared inputs, selected-serving
receipts and base shards together. `prepare` runs on CPU and freezes the resolved
profile, source snapshot, teacher alignment, augmentation audit when enabled,
token audit, ordered workloads and available trajectory lineage. Cached rendered
workloads must match their canonical inputs and tokenizer assets. Lineage checks
cover supplied trajectory hashes; absent grouping lineage cannot be inferred.
The initializer's file SHA256 is distinct from `model.initial_tensor_sha256`,
the training engine's ordered tensor fingerprint.

To start training before waiting for later stages, use `run --through train`.
Continue with `run --resume`. Completed stages are skipped only after checking
their input/source binding and every saved output checksum, including adapter
weights and merged shards. A lock prevents simultaneous drivers for one campaign.
An interrupted or failed stage retains its receipt/log and requires investigation;
`--resume` refuses to silently repeat a failed epoch or overwrite partial outputs.
Completed GPU stages release their models; the successfully evaluated scorer
stays warm for further use. The runner can retire the recorded previous scorer
before training, and retires its candidate scorer if evaluation fails.

`status` reads the driver receipt. `summarize` recomputes registered metrics from
saved predictions after verifying score responses and input metadata. `registry`
lists available evaluation groups. ID uses source-macro and pooled ranking plus
calibration diagnostics; APPS uses the existing clean-only operating point,
matched injection controls and raw partial AUC convention. New-adapter parity
limits are explicit and do not inherit a historical finite-quality waiver.
An explicitly authorized `evaluation.failed_parity_diagnostic: true` contract
may exceed the MAE limit while preserving failed gate values and retaining
finite/correlation/effect checks. Its summaries are labeled diagnostic; normal
evaluation still stops on failed agreement.
Outputs live under `results/<campaign_id>/`, data bindings under
`data/<campaign_id>/`, logs under `logs/runpod/<campaign_id>/`; runtime executables
and merged destination are configuration fields.

The training engine and selected serving migration helpers remain the validated
existing implementations. New campaigns need configuration rather than copied
training/evaluation controllers. The historical augmented replication retains
its original frozen controller and source binding.

Validation uses mocked stage launches, CPU contract checks and the current
artifact population. No second training run is required to test orchestration. The implementation
passed 58 focused checks; CPU preparation of the real frozen population bound
93 prerequisite artifacts and 307 source files. Its receipt and source snapshot
are in `results/monitoring-augmented-fp4-validation03/`; GPU stage launches were
mocked for the initial runner. Its shared completion validator also passed the
real completed 272-update metadata. The authorized augmented diagnostic uses
the shared optimized evaluator; see its [finding](../../docs/findings/b200_augmented_training.md).
