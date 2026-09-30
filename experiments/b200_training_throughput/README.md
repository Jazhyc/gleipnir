# B200 training throughput: ten-update iteration

Hypothesis: the B200's larger memory permits less activation recomputation or
larger length-grouped microbatches and improves on the optimized H100 recipe.
This systems screen uses the released 4B model's 21,837-row mixed training
population and its existing Kimi soft targets. A deterministic dataset/label
stratified 320-row sample is fixed across all conditions. Ten batches means
ten optimizer updates at effective batch 32, not a ten-microstep run or 10%
of the dataset. No ID/OOD evaluation or checkpoint selection is performed.
The deception portion lacks cached token lengths and some monitoring-specific
provenance fields. Preparation infers missing lengths with the pinned 4B
tokenizer solely for longest-row selection, records that operation in the
manifest, and preserves available provenance without inventing missing fields.
The original checksummed training prompts, labels, and targets stay unchanged.

The control retains rank-128 NF4/double-quantized/BF16 QLoRA, microbatch 1,
accumulation 32, selected-position direct logits, standard AdamW, FLA 0.5.2,
causal-conv1d 1.6.2.post1, Triton 3.7.1, linear-layer checkpointing, and the
H100-selected full-attention/linear-shell compilation policy. Global
torch.compile stays disabled; custom recurrent kernels remain eager.
The alternatives remove checkpointing at microbatch 1 or use length grouping
at microbatch 4 with accumulation 8. All see exactly the same 320 examples,
seed, objective, LR, rank, and effective batch. Grouping changes order and
padding; its effect is a combined batching intervention.

Run the control first after a one-update longest-32 memory/kernel preflight
and a same-weights compile canary. Stop on checksum drift, OOM, kernel/backend
failure, nonfinite loss/timing, graph explosion, or wrong update count. An
invalid candidate is retained as a failed systems result, not silently altered.
Report cold-inclusive Trainer throughput, warmup-excluded optimizer step times,
padding, memory, and projected one-epoch time for all 21,837 rows. Prefer gains
of at least 5%; a ten-update result is provisional and does not establish
model-quality equivalence or uncached production throughput.

During the first run, the user questioned the excessive startup delay. The
control's first update took 855.69 seconds with empty compiler caches. Before
either candidate started, their Triton and Inductor cache directories were
linked to the control's persistent cache on the same GPU/software stack. The
operational intervention is recorded in `runtime_cache_reuse.json`. This
reuses compatible kernels while allowing changed graphs to compile normally.
Cold-inclusive control/candidate throughput is consequently confounded by
cache state: the shared runner's automatic cold-throughput selection must not
be treated as the recommendation. Compare warm update times and verify the
selected recipe with a cached repeat, preserving the initial cold measurements.

The user authorized optimization on existing B200 Pod `alzfug70g5237b`, at
$6.79/hour, leaving it running afterward. Reuse persistent pinned kernels;
no additional billable capacity is launched. Logs go to
`logs/runpod/b200_training_throughput/`, and artifacts to
`results/b200_training_throughput/`. Keep FP32 benchmark adapters as masters;
they are systems artifacts, not released trained models.

```bash
.venv/bin/python -m gleipnir.monitoring_systems_screen prepare \
  --config experiments/b200_training_throughput/config.yaml
.venv/bin/python -m gleipnir.monitoring_systems_screen run \
  --config results/b200_training_throughput/resolved_config.json
```

Prepare on the Pod so the frozen manifest records remote absolute paths.
The launch script redirects logs and links the shared runner's default isolated
kernel targets to the already verified persistent installs. Initial ETA for a
single ten-update training loop is minutes; loading, tokenization, compilation,
preflight, and other conditions are separate. Agent startup checks stay active
in this turn. No after-turn scheduling capability has been verified.
