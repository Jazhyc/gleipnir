# Boundary-safe Qwen3.5 sequence packing

Date: 2026-10-02. The initial source/CPU audit preceded the authorized B200 work.
Subsequent BF16 packing canaries and matched eager/compiled training comparisons
completed, including a no-checkpoint follow-up. See the
[BF16 finding](../findings/bf16_sequence_packing.md) for results and limits.

## Prior evidence

The July 28 Phoenix 6.2 competition-NDIF packing experiment supplied reset
positions, cumulative sequence lengths and convolution sequence IDs, but only
2/96 scores agreed within 1e-5. Mean absolute probability drift was 0.12111,
maximum drift 0.52512, and macro AUROC fell from 0.99524 to 0.97619. Its small
runtime gain was rejected. This is evidence of a failed hosted integration,
not localization of a bug in today's B200 kernels. Original notes:
`Aletheias-Quest-Competition/experiments/phoenix_ndif_packing/README.md`, commit
`d28b3d05a7c0b19d9e17ca8c24603044eebaf737`.

Current adaptive batching keeps prompts in independent batch rows, so that
experiment does not demonstrate cross-example leakage in today's training.
The separate [B200 padding-mask diagnostic](../findings/b200_adaptive_microbatching.md)
found shape/mask-dependent gradients but did not prove leakage or an upstream
masking bug. Preserve those distinctions.

## Source audit

The local environment pins Transformers 5.14.1 and Torch 2.11.0. Audit the actual
installed B200 files against those identities before applying this design.

| Component | Existing mechanism | Required integration |
| --- | --- | --- |
| Full attention | `create_causal_mask` detects reset text positions when no padding mask or cache is supplied | Explicit block-diagonal causal isolation for a short oracle; variable-length attention for production |
| Gated DeltaNet | Model forwards `cu_seq_lens_q` as kernel `cu_seqlens` | Cumulative boundaries must survive every wrapper, checkpoint and backward |
| Causal convolution | Model forwards `seq_idx` to `causal_conv1d_fn` | Only same-example convolution taps may contribute; native compatible layout required |
| Monitoring readout | Existing trainer selects the last real position in each batch row | Select every original prompt's final input position in the flattened row |
| Loss/update | Existing adaptive trainer weights physical means by example count | Weight packed means by original example count, retaining logical update membership |

In installed `transformers/models/qwen3_5/modeling_qwen3_5.py`:

- `Qwen3_5TextModel.forward` accepts a mask dictionary with separate
  `full_attention` and `linear_attention` entries, and forwards kernel kwargs to
  decoder layers. Two-dimensional `position_ids` are expanded to four axes;
  the first axis carries the text positions used by attention-mask creation.
- `Qwen3_5GatedDeltaNet.forward` already passes convolution `seq_idx` and scan
  `cu_seq_lens_q`. No invasive model rewrite is initially required.
- `torch_chunk_gated_delta_rule` accepts `**kwargs` but does not use
  `cu_seqlens`. Its single running state crosses packed examples.
- The fallback convolution is ordinary `F.silu(self.conv1d(...))` over the
  flattened row. It does not consume `seq_idx`.

In installed `transformers/masking_utils.py`, packed attention detection runs
only when `attention_mask is None` and `past_key_values is None`. Supplying an
all-ones 2D padding mask bypasses that detection. A causal mask alone then lets
B attend to earlier A tokens. `create_recurrent_attention_mask` is a padding
signal, not a recurrent reset: it cannot establish sequence independence.
Do not pass a full-attention 4D mask to the recurrent padding-state multiplier.

The pinned [FLA 0.5.2 gated-delta entrypoint](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk.py)
propagates cumulative sequence lengths through forward and backward scan,
chunk construction and cumulative gate calculations. Its variable-length API
uses physical batch one. Separate logical examples remain separate scans.

Our [FlashQLA adapter](../../src/gleipnir/training/backends/flashqla.py) forwards
`cu_seqlens` through both the backend adapter and precision boundary. The
[pinned FlashQLA implementation](https://github.com/QwenLM/FlashQLA/blob/da06429d54b0f577de0a638f451ac8f0b395e0ac/flash_qla/ops/gated_delta_rule/chunk/__init__.py)
saves boundaries for backward and exposes automatic intra-card splitting. That
is source support, not proof of numerical isolation for our actual recipe.

[causal-conv1d v1.6.2 forward](https://github.com/Dao-AILab/causal-conv1d/blob/v1.6.2/csrc/causal_conv1d_fwd.cu)
and [backward](https://github.com/Dao-AILab/causal-conv1d/blob/v1.6.2/csrc/causal_conv1d_bwd.cu)
include sequence-ID comparisons in their channel-last paths. This implements
boundary-aware taps and gradients rather than a dense attention mask. The
repository installs 1.6.2.post1; audit that installed source/build independently,
including its channel-last/stride guards. Qwen's projected `[B,T,C]` tensor
transposed to `[B,C,T]` normally provides channel-last strides; wrappers must not
silently change them. Boundary IDs and cumulative lengths must be contiguous
int32 tensors on the operand device.

## Mask and state semantics

For each token t define a unique packed example ID s(t). Full-attention query q
may attend to key k exactly when `s(q) == s(k)` and `k <= q`. For Torch SDPA,
boolean True means allowed; a floating additive mask uses zero for allowed and
negative infinity for blocked. Do not interchange those conventions.

For width-four causal convolution, every contributing preceding tap must have
the same example ID as its output token. Example IDs must describe contiguous
runs without reuse. A boundary can occur at any offset, not only a kernel tile
boundary. Preserve these comparisons in dX and dW as well as forward.

The gated-delta state must begin independently at each cumulative boundary.
Masking its output or zeroing a boundary token does not erase the preceding
state. Gradients through the state recurrence and gate cumulative sums must
also stop at the boundary. Initially prohibit caches, persistent initial states
and final-state reuse for the packed training path.

## Concrete implementation route

1. Use one authoritative length list to build reset positions, int32 sequence
   IDs, cumulative offsets, maximum original sequence lengths, and all decision
   readout positions. Reject empty examples, inconsistent IDs, overflows and
   cross-device metadata. `PackedSequenceLayout` implements the CPU contract.
2. Keep packed batches free of a 2D padding mask. For short correctness probes,
   pass `attention_mask={"full_attention": block_causal_bool,
   "linear_attention": None}` and all kernel boundaries explicitly. This avoids
   relying on inferred full-attention segmentation.
3. Require the verified native FLA/FlashQLA and causal-conv1d callables. Reject
   reference fallbacks before packing; accepting a boundary kwarg in a Python
   signature does not demonstrate that an implementation honors it. If a
   reference fallback is needed for an oracle, split each operator by sequence
   and concatenate its outputs, as in the CPU diagnostic.
4. For long rows, use variable-length full attention with the same cumulative
   lengths. The existing FA4 interface is an integration candidate, even though
   its earlier padded-batch benchmark did not improve speed. A segmented SDPA
   reference is also valid but can incur per-sequence launch overhead. Dense
   masks compute/store quadratic packed-row work and are not the production
   performance design. Do not choose a backend solely from token counts.
5. Extend the direct monitoring readout to `cu_seq_lens_q[1:] - 1`, projecting
   only original prompt decision positions, and return scores in original
   logical-example order. Update physical-loss weighting to count examples,
   not the single flattened row. Auxiliary objectives remain unsupported until
   they have an explicit boundary contract.
6. Reuse existing checkpoint/selective-compilation policies only after their
   recomputation path passes boundary audits. Preserve the chosen NF4/BF16
   recipe and FlashQLA/FLA precision policy in each matched arm. Do not combine
   this with dtype, adapter, checkpoint or optimizer changes.

## Isolation gate before throughput

The [CPU experiment](../../experiments/monitoring_sequence_packing/README.md)
has deliberate leaking controls plus automatic and explicit attention-mask
positive controls, using segmented float32 CPU convolution and scan oracles.
Its results establish model-level routing semantics, not native B200 behavior
or the cause of the historical NDIF drift.

B200 tests should compare packed B with standalone B, perturb A while retaining
pack lengths, reorder examples, and differentiate a B-only scalar with respect
to input embeddings for A. Input gradients localize dependency; shared LoRA
parameter gradients should naturally sum across examples. Compare layer outputs
to find the first leakage before editing kernels. Test boundaries before/after
width-four convolution and 64/128-token chunk edges, length-one examples, and
actual short/long training inputs. Repeat with checkpointing/compilation.

If a native path fails, first verify boundaries actually reach its callable and
its bound implementation is the expected native kernel. Patch only a demonstrated
failure: convolution comparisons, per-example scan initialization, scan-local
gate sums, backward state boundaries or the full-attention routing. Keep patches
in an isolated pinned target with before/after hashes and CPU/GPU controls.
Do not patch all kernels on the assumption that historical NDIF leakage identifies
today's failing component.

## Completed CPU probe

The seed-0 `[7,5]` probe on the installed Transformers 5.14.1/Torch 2.11.0
float32 random-weight model completed. Both fully isolated controls produced
exactly zero packed-versus-standalone hidden-state difference, zero B-output
change when A was perturbed, and zero B-loss input gradient on A. Within-B input
gradients were nonzero, ruling out a disconnected backward calculation.

| Deliberate missing isolation | Maximum B hidden-state change after perturbing A | Maximum B-loss input gradient on A |
| --- | ---: | ---: |
| Both CPU kernel resets absent | 0.114408 | 0.216161 |
| Recurrent reset absent | 0.071332 | 0.036790 |
| Convolution reset absent | 0.055330 | 0.221075 |
| Both kernel resets present, all-ones 2D attention mask | 0.844604 | 3.777980 |

These values are synthetic hidden-state/input-gradient diagnostics, not
probability drift or quality metrics. They demonstrate three separate leakage
routes and show that a padding mask is not a substitute for packed isolation.
The local probe's package-discovery startup was accelerated with a temporary
process-local importlib metadata inference helper that groups candidate module
names before checking existing files; installed dependencies and model math
were unchanged. The checked-in entrypoint uses ordinary imports.

The artifact is `results/monitoring_sequence_packing/cpu_isolation.json` and
records the exact model source hash. Focused layout/isolation and existing
FlashQLA-wrapper tests passed: 40 passed, four deliberately skipped unsupported
partial-layer/ten-step policy combinations. Isolation cases include lengths
`[7,5]`, `[1,3]` and `[63,65]`;
nonfinite negative controls also fail the audit. Ruff and diff checks passed.
No after-turn monitoring or automatic B200 launch is scheduled by this document.

## BF16 LoRA integration (October 2)

The user selected regular BF16 LoRA after the completed matched recipe comparison.
The opt-in B200 screen now uses that frozen BF16 base, FP32 rank-128 adapters,
uniform FlashQLA with FP32 normalization/gates and BF16 Q/K/V, and the existing
checkpoint/compile policies. It passes one explicit layout to positions, native
convolution and gated-delta recurrence. Full attention executes causal SDPA
separately for each segment, with the same compilation-opaque router installed
for the padded control. This avoids a quadratic dense mask over long packs.

The screen retains original example targets and equal example weighting, packs
only within the logical update, and selects one final-input-token projection
per example. The default training path remains padded. Identical-shape prefix
perturbation, checkpointed input-gradient isolation and matched singleton
monitoring-loss/adapter-gradient gates run before optimizer updates and again
after compilation. Tolerances and the bounded matched timing protocol are in
the [experiment README](../../experiments/monitoring_sequence_packing/README.md).
The real-model GPU isolation checks passed with zero measured cross-example
effects. Corrected compiled packing passed the 5% singleton/packed gradient gate
and reduced update time by 26.65% on the matched cohort. Removing model
checkpointing reduced packed update time a further 9.99%, fitting at 147.14 GiB
peak allocated memory. These bounded results support an opt-in systems recipe;
longer training and frozen held-out quality validation are still required.
The earlier compiler-related numerical discrepancies remain recorded with
unresolved cache-causality attribution in the finding.

Subsequent comparisons use packing only by user instruction. Doubling the packed
token budget to 32,768 with checkpointing retained reduced physical calls from
74 to 44, but improved time only 3.50% (5.5162 seconds/update). It remained 7.21%
slower than packing without checkpointing, at lower memory (111.04 versus
147.14 GiB). This tested batch-size intervention did not beat removing
recomputation; the optimal larger budget has not been established.
