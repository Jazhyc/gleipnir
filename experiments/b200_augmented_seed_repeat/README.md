# Same-seed BF16/FA4 augmented training repeat

Repeat the completed `b200-augmented-bf16-training02` seed-0 recipe to measure
execution variability. Keep its original zero-B rank-128/alpha-256 FP32
initializer, 8,688 augmented rows (3,475 injected), soft teacher targets,
one epoch/272 logical updates, logical batch 32, AdamW 5e-5, linear decay/3%
warmup, zero dropout/weight decay/checkpointing, BF16 MLPs, BF16 FlashQLA,
packed FA4 4.0.0b33 and selective compilation. Reuse the pinned runtime,
startup validation and persistent caches. Do not enable new deterministic
controls: this measures the existing recipe's repeat variability.

Before launch, compare the resolved job with the completed seed-0 job; only
output/log paths and run identity may differ. Check executed training source
identity. After training, compare all 4,556 physical batches, update/data order,
token totals, runtime, losses and final FP32 adapter tensors. Preserve both
adapters. Differences do not by themselves establish which kernel caused them.

Select only the final complete 272-update checkpoint. Merge in FP32 then export
BF16 to its own ephemeral checkpoint. Require fresh master/merged/BF16 serving
parity (MAE <=0.020, correlation >=0.99, nonzero effect and finite scores) and
native BF16 audits. Score the same 3,012 CoT-removed ID examples/33,750,959 input
tokens on unquantized BF16 vLLM 0.31, native tokenizer, cached causal LAST
0/1 head, corrected scheduler, prefix off, original 128-row groups and c128.
Use the completed seed-0 BF16/FA4 ID predictions as the primary paired control.
Report macro/pooled/source AUROC and raw pAUROC@20, calibration, ties,
threshold diagnostics, paired score/decision shifts and input throughput/latency.
One repeat provides an observed difference, not a reliable variance estimate.
No data-order seed change, historical SDPA retraining, APPS or promotion.

Stop on recipe/identity/runtime drift, nonfinite/missing gradients or scores,
OOM, failed adapter/native gates, incomplete coverage or completion. Retire the
current historical diagnostic scorer for training after CPU checks. Keep model
masters, measured baselines and persistent caches. Checksum-archive the inactive
previous BF16/FA4 merge to the volume to make ephemeral space; preserve its
restore path and receipts. Use the existing authorized NC2 B200 without a
capacity change. No in-chat heartbeat tool is available; monitoring is active
only during the turn.

Entrypoint: `python -m gleipnir.campaigns.monitoring prepare --config
experiments/b200_augmented_seed_repeat/config.yaml`, then `run` with the same
config in the preserved training runtime. The existing campaign runner performs
training, reference, merge and BF16 ID stages. Artifacts use
`b200-augmented-bf16-seed0-repeat01`; the merged checkpoint is
`/tmp/gleipnir-merged/bf16-augmented-fa4-seed0-repeat01`.
