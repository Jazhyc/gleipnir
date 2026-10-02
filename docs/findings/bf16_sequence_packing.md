# BF16 LoRA sequence packing on B200

Status: native eager isolation/parity and matched timing passed; an explicitly
authorized compiled learning diagnostic is under validation. Packing remains
opt-in; ordinary training uses adaptive padding.

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

## Artifacts and limits

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
processed tokens, so fewer physical calls also contribute to the gain. Memory
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
The corrected protocol writes `results/bf16_sequence_packing_checked/`, including
separate padded/packed reports and FP32 masters on successful completion. Setup
and compilation are outside synchronized measured training steps. The baseline
is the same BF16/backend/reduction recipe in the same process, with fresh AdamW
states and ten identical logical updates per condition.

The eager canaries use two training examples truncated for bounded kernel tests;
their loss/gradient agreement is a systems check, not task-quality evidence.
There is no held-out quality result. Full-campaign promotion still requires the
frozen validation contract. Focused regressions passed: 108 tests, eight intentional unsupported
policy skips. The [experiment README](../../experiments/monitoring_sequence_packing/README.md)
records hypotheses, gates, stop conditions, and executable configurations.
