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
control's first update took 855.53 seconds with empty compiler caches. Before
either candidate started, their Triton and Inductor cache directories were
linked to the control's persistent cache on the same GPU/software stack. The
operational intervention is recorded in `runtime_cache_reuse.json`. This
reuses compatible kernels while allowing changed graphs to compile normally.
Cold-inclusive control/candidate throughput is consequently confounded by
cache state: the shared runner's automatic cold-throughput selection must not
be treated as the recommendation. Compare warm update times and verify the
selected recipe with a cached repeat, preserving the initial cold measurements.
For that verification, freeze a second job manifest from every successfully
validated condition, retaining the same names, 320-row selection, seed, batch,
rank, objective, and training settings while assigning separate output/log
directories. Run those jobs serially after the original screen finishes, using
the shared warmed cache. Compare the complete ten-update Trainer runtimes;
excluding the first two updates can bias length-grouped cases by dropping their
longest batches. A failed original condition is recorded and excluded from
recommendation rather than rerun with altered settings.

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

The initial control completed, but full checkpoint removal exhausted GPU memory
after one update (176.05 GiB allocated; 178.00 GiB process use). The shared runner
stopped the campaign; microbatch 4 was never launched. Preserve that failed
campaign. `warm_config.yaml` defines a separate continuation with the same fixed
sample: a warmed control, checkpoints on 12 of the 24 linear layers, and grouped
microbatch 2 with accumulation 16. Preflight the half-checkpoint recipe on the
longest 32 rows, including the existing same-weights numerical canary. Use
`warm_launch.sh` with the synced commit in `GLEIPNIR_COMMIT`; it records cache
reuse before launching. Keep the 5% gain gate. If the larger batch looks promising,
repeat it with its new graphs cached before recommending it. Compare complete
loops rather than dropping their longest batches as warmup.

The FA4 intervention uses `fa4_config.yaml`: matched SDPA and
`flash_attention_4` conditions at microbatch 1, accumulation 32, retaining all
24 linear checkpoints. The eight full-attention layers change backend; the
FLA layers, prompts, objective, sample, and compilation policy stay fixed.
Pin FA4 4.0.0b33 and its CuTe dependencies in an isolated overlay using
`fa4_bootstrap.sh`, preserving the locked serving environment. Both conditions
use that same overlay and persistent compiler cache. Before timing, require
BF16 GQA forward/backward agreement with FP32 math attention at the actual
model head dimensions, then same-model SDPA/FA4 decision-logit agreement on
two unequal-length padded sequences, the compile canary, and the longest-32
training update. Existing logit tolerances are 0.05 absolute plus 0.01 relative;
the kernel probe requires relative L2 error <=0.02 forward and <=0.05 backward.
Stop on any failed check or existing failure criterion. Use the 5% gain gate;
repeat a promising FA4 timing with its new kernels cached. Record cold FA4
startup separately from steady performance. No inference backend changes.
The launcher explicitly enables FA4's opt-in persistent CuTe kernel cache on
the network volume; its default is an in-process cache only.

The grouped batch-2 first pass completed with 5.22% padding and 142.51 GiB
peak allocation, but compilation still inflated many of its ten updates.
`batching_repeat_config.yaml` repeats the matched control and grouped batch 2
with those kernels cached and performs the longest-32 preflight for batch 2.
Use the same FA4 overlay as the preceding attention comparison, explicitly
requesting SDPA for both batching conditions. Keep the same sample and 5% gate.
Only a complete warmed loop and successful longest-row preflight support
recommending batch 2; its isolated fast updates are insufficient.

The first FA4 preflight completed its longest-row update and numerical canaries
but failed the frozen limit with 32 Dynamo graphs; neither timing condition ran.
Logs identify static `module.layer_idx` specialization across full-attention
layers. `fa4_dynamic_config.yaml` is a separate follow-up, enabling Torch's
`allow_unspec_int_on_nn_module` for both SDPA and FA4. Keep the same 24-graph
bound and all correctness gates. This is a matched compiler intervention,
not a relaxed acceptance criterion. Use `fa4_dynamic_launch.sh` after the
failed campaign ends; it retains the failed artifacts and verified kernel probe.

The dynamic-integer follow-up completed its update but produced 44 graphs:
it removed index specialization while exposing FA4's Python kernel interface
to more tracing. It also failed before timing conditions began.
`fa4_interface_config.yaml` instead keeps the selected attention interface
outside Torch compilation for both SDPA and FA4, preserving compiled surrounding
layers and native GPU kernels. Dynamic module integers return to their original
default. Retain the 24-graph limit, same-weight backend and compilation canaries,
longest-row update, fixed cohort, and 5% gain gate. Use
`fa4_interface_launch.sh`; neither failed integration attempt is recommended.

The opaque-interface comparison passed and completed both ten-update loops:
SDPA 366.58 seconds versus FA4 365.80 seconds, a 0.21% loop-time difference.
FA4's last-eight mean was 20.67 seconds versus SDPA's 20.10 seconds. This falls
below the 5% gain gate and does not justify an additional promising-candidate
repeat or backend switch. Keep SDPA; native FA4 correctness is established for
these bounded checks, but no training-speed benefit was measured.
