# BF16 LoRA sequence packing on B200

Status: eager and corrected compiled packing passed isolation, packing parity
and bounded learning/timing checks, including the no-checkpoint follow-up. Packing remains
confined to the bounded screen; new comparisons use packing only by user
instruction. Historical paired reports remain available as baselines.

## Matched recipe and boundaries

The user selected regular BF16 LoRA. This screen uses the pinned Qwen3.5-4B
revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, a fully frozen BF16 base,
169,869,312 trainable FP32 rank-128 adapter parameters, the existing initial
adapter digest, and the existing hashed stratified 320-example mixed cohort.
The full source dataset contains 21,837 rows. No new teacher requests, truncation
policy, held-out selection, or final-test promotion is involved.

Packing is best-fit within the same logical update of 32 equally weighted
examples, with a 16,384-token budget and intact oversized singletons. Every
original final-input-token decision and soft target is retained. The native
convolution receives `seq_idx`; FlashQLA receives cumulative sequence lengths;
RoPE positions reset per example. Full attention runs independent causal SDPA
calls per segment. No dense long-context mask or 2D padding mask is supplied to
the packed row. Both timing controls use the same compilation-opaque SDPA router.

## Isolation versus numerical drift

The first native BF16 attempt passed identical-shape prefix perturbations and
checkpointed cross-example input-gradient checks at lengths `[1,3]`, `[63,65]`
and `[127,129]`. The retained second-run receipts show exactly zero score drift
and cross-example gradients, with nonzero own-example gradients in every case.
Nevertheless, the packed/singleton adapter-gradient relative L2 was 0.0581823,
above the predeclared 0.05 gate. Its losses were 0.939848125 and 0.957633674.
No optimizer updates ran.

Disabling intermediate BF16 GEMM reductions alone reproduced those values
exactly. Layer traces showed an exact first input normalization, QKV projection
and recurrent-mixer match. The first difference occurred in the first
MLP/residual output (maximum absolute 0.00048828125) and grew through depth.
This was numerical disagreement despite demonstrated sequence independence.

The working eager configuration explicitly selects cuBLASLt and sets
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)`.
The second boolean disables split-K reductions; boolean `False` alone retains
them. The installed Torch build requires cuBLASLt for no-split-K GEMMs. The first
attempt without that backend stopped at a runtime prerequisite with zero updates.
These controls change GEMM execution, preserving BF16 parameters/activations and
FP32 adapter masters. They are applied identically to the padded and packed
controls. See the [PyTorch numerical-accuracy documentation](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).

With that combined backend/reduction configuration, the real-model eager probe
matched every recorded layer output and the final projection exactly. Both
monitoring losses were 0.949228942. Adapter-gradient relative L2 was 0.008881947,
below the unchanged 0.05 gate. All prefix perturbation and cross-example input
gradients remained exactly zero. This evidence establishes the combined
configuration's effect; it does not isolate the contribution of backend selection
from disabling split-K.

Temporary forward hooks used for layer traces proved unsuitable inside compiled
graphs: a cached graph bypassed their collection and the diagnostic stopped on
a missing entry, before updates. Layer traces are now eager-only. Compiled score,
loss, checkpointed input-gradient and adapter-gradient gates remain unchanged.
The corrected protocol is `experiments/monitoring_sequence_packing/bf16_checked_gpu.yaml`.

That protocol passed eager isolation/parity again. Its compiled isolation checks
also retained zero cross-example dependence, but singleton/packed losses were
0.949228942/0.957633674 and adapter-gradient relative L2 was 0.0880173. It stopped
before updates. The next matched condition,
`experiments/monitoring_sequence_packing/bf16_casts_gpu.yaml`, enables Inductor's
intermediate precision-cast emulation in both controls. Installed Torch's
`_inductor/config.py` documents that ordinary fusion elides low-precision
downcast/upcast pairs, whereas this option preserves eager rounding boundaries.
This intervention does not relax any gate or establish quality equivalence.

Cast emulation did not resolve compiled numerical parity: singleton/packed losses
were 0.949228942/0.967125356 and gradient relative L2 was 0.143217. Sequence
isolation still passed, and no updates ran. Compiled packing is therefore rejected
under the fixed 0.05 gate for both tested compiler precision policies. The
validated native eager mode is compared with matched eager adaptive padding in
`experiments/monitoring_sequence_packing/bf16_eager_gpu.yaml`. Both controls retain
the same checkpoint policy and native FlashQLA/convolution kernels, with Torch
model compilation disabled. Any speed result from that condition is an eager
comparison, not a comparison with the original compiled BF16 recipe.

The user subsequently authorized testing short-term training stability despite
the 14.3% gradient disagreement. `bf16_learning_gpu.yaml` explicitly permits
gradient relative L2 up to 0.15 for the bounded ten-step diagnostic, preserving
the strict 0.05 parity failure and recording learning acceptance separately.
Isolation, finite-value and loss-agreement gates remain unchanged. Matched
initial adapters, example order, targets, optimizer and common-backend probes
permit a trajectory comparison; longer training and held-out validation are
still required before promotion.

The first learning attempt accepted the exact same 0.143217 compiled gradient
result. Its boundary-canary variants then exhausted Torch's default eight-graph
recompile limit during longest-example preflight; a linear decoder shell fell
back to eager because the keyword count differed. It was interrupted before
optimizer updates, retaining a separate interruption receipt. The corrected
`bf16_learning_cached_gpu.yaml` explicitly raises the shared limit to 64 and
enables `fail_on_recompile_limit_hit`, so this diagnostic cannot silently time
an eager fallback after cache exhaustion. This changes compiler cache policy
equally for both controls, preserving model, precision and learning settings.

The corrected run's initial compiled canary passed strict parity: both losses
were 0.957633674 and gradient relative L2 was 0.00885904. This is an observed
improvement in singleton/packed agreement after the cache intervention, not
proof of improved accuracy against eager or a localized explanation of the
old discrepancy. The earlier run's first explicit cache-exhaustion warning
appears after its numerical canary, and the generated continuation-graph traces
differ between runs. Identical generic compile-canary losses (eager 1.287744,
compiled 1.320180) also show that ordinary compiled/eager disagreement remains.
Root-cause attribution needs controlled graph/kernel replay; cache size alone
is not a numerical-accuracy setting. Torch's [recompilation documentation](https://docs.pytorch.org/docs/2.11/user_guide/torch_compiler/compile/programming_model.recompilation.html)
explains the compile-versus-eager fallback behavior.

## Artifacts and limits

The corrected compiled comparison completed ten steps per control with no
cache-fallback warning and zero additional Dynamo graphs during measured steps.
Both initial adapter digests, all dataset indices, learning rates and actual
token counts matched. The shared padded/original-backend probe began at exactly
1.203672633 in both conditions; all losses/gradient norms remained finite and
both FP32 master digests changed.

| Compiled control with checkpointing | Adaptive padding | Packing |
| --- | ---: | ---: |
| Mean synchronized update seconds | 7.7928 | 5.7163 |
| Physical forward/backward calls | 115 | 74 |
| Peak allocated memory, GiB | 98.0445 | 98.0446 |
| Final shared-probe loss | 0.77854 | 0.77345 |

Packing reduced update time by 26.65% (36.32% greater throughput). The maximum
absolute difference between stepwise mean probe losses was 0.03025, with final
difference 0.00509. The gradient norms were 3.77–27.46 padded and 3.93–27.17 packed.
This supports early learning stability and systems efficiency on this cohort;
it does not establish longer convergence or held-out task-quality equivalence.
Although the 15% diagnostic allowance remained configured, both packing
canaries passed the original 5% gate, so this trajectory does not directly
demonstrate training stability with a 14.3% canary disagreement.

The user requested a follow-up without gradient checkpointing.
`bf16_no_checkpoint_gpu.yaml` retains the compiled recipe and both layouts,
changing only model checkpointing and output destinations. The first test is
the unchanged longest-32 preflight; an OOM stops the comparison without shrinking
its token budget or examples. Setup and measured memory/time will be compared
with the table above.

The no-checkpoint comparison completed with no checkpointed layers in either
model receipt, no cache-fallback warnings and no new measured Dynamo graphs.
Initial adapter hashes, all batches, learning rates, physical partitions and
common-probe starting values matched the checkpointed runs. Both longest-32
preflights passed. Compiled singleton/packed canary losses were equal at
0.949228942; gradient relative L2 was 0.00697903. All tested cross-example logit
effects and input gradients were exactly zero, and every training loss and
gradient norm was finite.

| Compiled control without checkpointing | Adaptive padding | Packing |
| --- | ---: | ---: |
| Mean synchronized update seconds | 7.1180 | 5.1452 |
| Peak allocated memory, GiB | 147.1394 | 147.1395 |
| Final shared-probe loss | 0.78949 | 0.77271 |

Disabling checkpointing reduced padded update time by 8.66% and packed update
time by 9.99%, exceeding the prospective 5% gate for both. It used approximately
49.10 GiB more allocated memory, while the unchanged cohort still fit on the
B200. Packing alone reduced no-checkpoint update time by 27.71%; combining
packing and no checkpointing reduced time by 33.97% against the compiled padded
checkpointed control. The no-checkpoint packed probe ended within 0.00075 of
checkpointed packing's probe; this remains a small training-cohort diagnostic,
not held-out quality equivalence. The configuration is a supported opt-in
candidate for a longer validated campaign, not a default-recipe promotion.

## Larger packed batches with checkpointing

The user requested packing for all subsequent comparisons. The runner now
uses `packing_only: true` by default; historical paired configurations remain
explicit replays. The new `bf16_larger_batch_gpu.yaml` run retained the same
12 checkpointed layers and doubled the packed token budget to 32,768. Logical
batch size, initial adapters, all 320 examples/order, targets, learning rates,
optimizer and shared probe remained matched to the completed packed baselines.
No new padded training trajectory ran.

| Compiled packed recipe | Seconds/update | Peak allocated GiB | Physical calls |
| --- | ---: | ---: | ---: |
| 16,384 tokens, checkpointed | 5.7163 | 98.0446 | 74 |
| 16,384 tokens, no checkpointing | 5.1452 | 147.1395 | 74 |
| 32,768 tokens, checkpointed | 5.5162 | 111.0401 | 44 |

The larger checkpointed batch reduced calls by 40.54% and update time by 3.50%
against checkpointed packing at 16k, using 13.00 GiB more allocated memory.
It was 7.21% slower than the no-checkpoint packed run, while using 36.10 GiB
less allocated memory. It did not meet the prospective requirement to beat
no checkpointing by at least 5%. On this cohort and tested budget, spending
memory on removing recomputation gave the larger speed gain. This does not
establish the optimal token budget or exclude gains at other batch sizes.

The longest-32 preflight and all ten warmup/measured batches passed. The compiled
packing canary again had gradient relative L2 0.00885904, equal singleton/packed
losses and exactly zero measured leakage. All training losses/gradient norms
were finite, with no fallback warnings or new measured Dynamo graphs. Every
recipe processed 1,314,331 actual tokens; packing introduced no padding. Shared
probe losses started identically at 1.20367263 and ended at 0.77345017 (16k
checkpointed), 0.77270733 (16k no checkpointing) and 0.77203090 (32k checkpointed).
These are ten-step training-cohort diagnostics, not convergence or held-out
quality evidence. The new receipts, FP32 master, matched comparison and plot
are collected under `results/bf16_sequence_packing_larger_batch/`.

## Earlier eager comparison

The matched eager trajectories completed ten steps each (the first with zero
learning rate), preserving the same initial adapter digest, data order, learning
rates and 1,314,331 actual tokens. All losses and gradient norms were finite;
both final master digests changed.

| Eager control | Adaptive padding | Packing |
| --- | ---: | ---: |
| Mean synchronized update seconds | 9.3926 | 7.2430 |
| Physical forward/backward calls | 115 | 74 |
| Processed tokens | 1,411,298 | 1,314,331 |
| Peak allocated memory, GiB | 118.4226 | 118.4225 |
| Final original-backend probe loss | 0.79265 | 0.76925 |

Packing reduced measured update time by 22.89% (29.68% greater throughput),
exceeding the predeclared 5% speed gate. Padding already wasted only 6.87% of
processed tokens; the gain is consistent with fewer physical calls as well as
removed padding, without isolating their contributions. Memory
was effectively unchanged; intact long examples dominate the peak. Setup,
probes and artifact export are outside the synchronized update measurement.
This is one ten-step trajectory per condition, not repeated confidence evidence.

The eager probe used original kernels/forwards but each condition's collator.
Its initial losses differed slightly, 1.20367 versus 1.20243. Therefore the
displayed final probe losses support only a rough stability comparison, with
both trajectories improving; they are not identical-execution quality scores.
The compiled diagnostic now uses the original padded collator for both common
probes, eliminating this avoidable layout difference.

Earlier receipts/logs are retained locally under the corresponding
`results/bf16_sequence_packing*` and `logs/runpod/bf16_sequence_packing*` trees.
Completed reports and both FP32 masters per run are under
`results/bf16_sequence_packing_eager/`,
`results/bf16_sequence_packing_learning_cached/` and
`results/bf16_sequence_packing_no_checkpoint/`. Each has a configuration/source
contract, trajectory summary and plot; the no-checkpoint run additionally has
a matched checkpoint comparison. Locally collected artifacts are checksummed.
Earlier strict failures and the interrupted cache-exhaustion attempt remain
separate, preserving negative results. Setup
and compilation are outside synchronized measured training steps. The baseline
is the same BF16/backend/reduction recipe in the same process, with fresh AdamW
states and ten identical logical updates per condition.

The eager canaries use two training examples truncated for bounded kernel tests;
their loss/gradient agreement is a systems check, not task-quality evidence.
There is no held-out quality result. Full-campaign promotion still requires the
frozen validation contract. Focused regressions passed: 122 tests, eight intentional unsupported
policy skips. The [experiment README](../../experiments/monitoring_sequence_packing/README.md)
records hypotheses, gates, stop conditions, and executable configurations.
