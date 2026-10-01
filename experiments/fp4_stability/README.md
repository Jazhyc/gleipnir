# FP4 stability diagnostic

Hypothesis: the previous eager/compiled loss disagreement must be separated
from low-precision backward instability. First reproduce forward comparisons
with original BF16 MLP bases and native Four Over Six bases, repeated calls,
and prefixes of the existing compiled decoder policy. No optimizer updates in
the diagnostic stage. Keep the existing absolute 0.01 + 1% loss gate.

Then test BF16 input-gradient GEMMs using the BF16-dequantized exact forward
packed weight. Frozen bases do not need weight gradients; LoRA branches retain
FP32 master adapters. This adopts the dequantized-backward idea from
[The 4-bitter Lesson](https://humansand.ai/blog/nvfp4-rl), not its MoE/RL stack.
Native W4A4 forward GEMMs remain explicit CUTLASS, with MSE 4/6 block selection.
The original fully quantized backward remains an independently recorded control.
The decoded frozen-weight cache costs memory and is not a memory optimization.

An explicit follow-up option normalizes each token row by its FP32 maximum,
converts the normalized input to BF16, and quantizes with a fixed global maximum
of one, then rescales the BF16 GEMM result in FP32 and rounds it to BF16. This
unfused diagnostic removes cross-token scale dependence but is not the blog's
fused kernel or a claim of bit-exact TransformerEngine parity. Native decoded
arithmetic and an outlier-row independence canary must pass before model use.

Use the frozen Qwen3.5-4B revision, training inputs/Kimi targets, selected
320-example cohort and global longest-32 preflight from the B200 experiment.
Retain rank 128, alpha 256, AdamW 5e-5, logical batch 32, physical maximum 8,
16,384 padded-token budget, twelve checkpoints, SDPA, pinned FLA/convolution,
and the established selective compiler policy. Attention storage remains NF4
and compute BF16, as in the control. No generation KV cache or final test access.
Record actual hardware and cache state for every run. Match all comparison
conditions on the same GPU; do not transfer timings across GPU families.

Current target: original B200 Pod `alzfug70g5237b` in US-NC-2 at $6.79/hour,
with network volume `ixbh81vf9c`. Its preserved environment was verified after
resume: Python 3.12.3, Torch 2.11.0+cu130, GPU SM100, 183,359 MiB, zero volatile
uncorrectable ECC errors. Frozen inputs and kernel/model caches remain on the
volume; no data migration was required. Each campaign still verifies its own
input identities and native canaries. The infrastructure record documents the
[reservation handoff and unused stopped Pods](../../docs/infrastructure.md#runpod).

Stop on input drift, backend/kernel failure, OOM, nonfinite values, unchanged
adapters after updates, or failed numerical gates. Diagnostic loss comparisons
may record failures but never authorize training past them. No quality promotion
from short trajectories; separate grouped held-out validation remains required.
Inspect startup every 30–60 seconds. No in-chat scheduler is available, so
monitoring applies during the active turn only.

`row_aot_training.yaml` tests the bounded trajectory with `aot_eager` after
that backend matched eager exactly on the matched-cohort forward diagnostic.
Native FP4, BF16 and NF4 MLP conditions share this backend, initialization seed,
data, checkpointing and adaptive batching. Each first performs the global-longest
update, followed by ten matched updates. This tests optimizer stability while
Inductor numerical consistency remains unresolved; timings do not compare
against the original optimized Inductor recipe.

The first NF4 pass is a finite-training result but has different initial adapter
values despite the same seed. `row_aot_nf4_matched.yaml` repeats that condition
from an explicit standard PEFT initialization artifact whose complete tensor hash
equals the native/BF16 initial hash. It retains the global update and ten-step
stop condition. The runner records initialization file checksums and the screen
fails before GPU preflight if the loaded master hash differs. Same seed alone
does not establish matching initialization across dense/quantized module classes.

Completed outcome: native per-token W4A4 MLP forward plus decoded-weight BF16
backward passes the global-longest update and ten cohort steps with `aot_eager`.
BF16 and corrected NF4 controls also pass, with exact initial adapter, order,
batch and learning-rate matching. Cohort peak allocated memory is 135.2 GiB
native, 116.7 GiB BF16 and 134.8 GiB NF4. Native common training-probe loss falls
1.165299 -> 0.779105. Twenty focused tests pass. This establishes bounded LoRA
training viability; Inductor parity, activation-memory savings, matched speedup
and held-out quality remain unestablished. Full evidence and reproduction limits
are in [the finding](../../docs/findings/fp4_training_stability.md).

## FP4 timing replay

`row_aot_timing.yaml` benchmarks only the stable native FP4 configuration.
Hypothesis: the earlier first-pass times overstate repeated step cost because
of compilation and kernel-cache work. Retain all arithmetic, data, initial
adapter hash, batch policy, twelve checkpoints and AOT backend. There are no
additional precision controls or held-out evaluations in this benchmark.

After the usual native kernel, numerical and global-longest update gates, run
the entire frozen ten-batch cohort once as warm-up and three times as measured
replays in the same process. Restore the exact initial FP32 adapters and create
fresh AdamW state and a fresh ten-step scheduler for every pass. Each pass has
one LR-zero step and nine nonzero-LR updates. Do not select or discard slower
batches. Stop after forty cohort steps or on any existing failure condition.

Synchronize CUDA around each complete logical step, including forward,
backward, finite checks, clipping, optimizer and scheduler. Report every step,
warm-up and measured pass times, token-weighted throughput, batch-specific
means, memory and Dynamo graph counts. Report writing, restoration, model
loading, probes and checkpoint export are outside step timing. Loop wall time
also includes report writing. Compilation graphs must stop growing during
measured passes before treating them as warm. Existing disk caches are reused;
the first pass measures this process's warm-up, not a pristine-cache startup.
Native quantization and per-token scaling remain unfused in this prototype.

```bash
bash experiments/fp4_stability/launch.sh experiments/fp4_stability/row_aot_timing.yaml
```

Completed: the three measured pass means are 15.541/15.897/15.600 seconds per
step, with zero new Dynamo graphs in each. All thirty measured steps average
15.679 s (median 15.440 s), delivering 8,382.6 actual tokens/s on the frozen
cohort. The reused-cache warm-up pass takes 177.174 s of step work. Matching
initialization/workload and source hashes pass the collected analysis; final
adapter hashes differ across replays, so bitwise trajectory identity is not
claimed. See the [finding](../../docs/findings/fp4_training_stability.md#warmed-fp4-timing-benchmark)
for scope, memory, provenance and excluded setup costs.

The separate `precision_cast_diagnostic.yaml` retains the original FP4 scaling
and backward while setting `TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1`. The pinned
Torch source documents that default fusion can remove intermediate BF16
downcast/upcast pairs; the option preserves eager rounding boundaries. This is
a compiler hypothesis, not an established cause of the previous failure.
Ten-update campaigns automatically run a separate one-update global-longest-32
stage first for each precision condition, with zero warmup. Any failed stage
stops the campaign before the next stage.
`row_precision_cast_diagnostic.yaml` explicitly selects the matched 320 rows
with diagnostics only. Its ten-batch selection size never authorizes optimizer
updates or queues a global optimizer preflight. This checks input-dependent
compiler consistency after the first matched-cohort gate failure.

Initial implementation validation: 13 focused CPU tests passed using Torch
2.11.0+cpu and other exact locked dependencies in isolated `/tmp` overlays;
local CUDA/native dependency reads were stalling on Lustre. Ruff and shell syntax
checks passed. On the restored B200, original FP4 forward/backward canaries
passed at batch sizes 1/2/4/8. Per-token forward plus dequantized BF16 backward
also passed, with maximum relative L2 0.00235 forward and 0.00167 backward;
the outlier-row independence comparison had zero relative error. Artifacts are
`results/fp4_forward_diagnostic/kernel_canary.json` and
`results/fp4_native_stability/kernel_canary.json`. These are isolated arithmetic
checks; full-model numerical agreement and optimizer behavior are separate gates.

```bash
bash experiments/fp4_stability/launch.sh experiments/fp4_stability/config.yaml
```
