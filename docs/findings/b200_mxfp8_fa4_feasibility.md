# MXFP8 FlashAttention 4 integration feasibility

Date: 2026-10-04. Assessment of the user-linked
[PyTorch article](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)
and Meta's public `lp_fa4` implementation at revision
`2889aefba0d03b3eecf779eed9f4097066ecdc5d`.

## Finding

Blackwell hardware and our packed attention boundary are suitable integration
points, but the released MXFP8 forward/backward kernels cannot execute the
current Qwen3.5-4B recipe. This needs upstream or custom kernel development,
not a precision configuration change. Keep the user-selected BF16 FA4 default.
No CUDA kernels, model updates, package upgrades or infrastructure changes were
performed for this assessment. There is no measured MXFP8 speed or quality
result for Gleipnir.

| Requirement | Current recipe | Public MXFP8 path | Assessment |
| --- | --- | --- | --- |
| Hardware | B200, SM100, CUDA 13, Python 3.12 | Blackwell SM10x, CUDA 13, Python 3.12 | Suitable hardware |
| Mask | Causal self-attention | Noncausal only | Unsupported |
| Head dimension | Q/K/V 256 | Q/K/V 128 only | Unsupported |
| Heads | 16 query, 4 key/value; GQA ratio 4 | Equal query and key/value heads; MHA | Unsupported |
| Packing | Independent variable-length examples | Packed variable lengths with sequence-local scales | Compatible conceptually |
| Runtime | Torch 2.11.0, CUTLASS DSL 4.8.0, pinned TVM FFI 0.1.11 | Tested constraints pin Torch 2.13.0, CUTLASS DSL 4.6.1, TVM FFI 0.1.13.post3, Quack 0.6.2 | Requires isolated compatibility work |
| Autograd | Native BF16 FA4 wrapper in packed router | Explicit MXFP8 forward and backward functions | Needs a training wrapper |

The last row describes the public MXFP8 API, not the separate BF16 API, which
already supplies an autograd interface. The runtime row describes the tested
upstream environment, not proof that every older dependency is incompatible.
Our pinned FlashQLA dependency precedence must also survive any combined-process
integration.

## Evidence and bounded checks

The upstream [README](https://github.com/facebookresearch/ads_model_kernel_library/blob/2889aefba0d03b3eecf779eed9f4097066ecdc5d/lp_fa4/README.md)
states the initial noncausal D128 MHA scope. Inspection confirms actual guards:

- [`mxfp8.py`](https://github.com/facebookresearch/ads_model_kernel_library/blob/2889aefba0d03b3eecf779eed9f4097066ecdc5d/lp_fa4/src/lp_fa4/cute/mxfp8.py):
  quantization rejects source shapes other than `(tokens, heads, 128)`;
  forward validates all operands against the query head count and exposes no
  causal parameter.
- [`interface.py`](https://github.com/facebookresearch/ads_model_kernel_library/blob/2889aefba0d03b3eecf779eed9f4097066ecdc5d/lp_fa4/src/lp_fa4/cute/interface.py):
  MXFP8 forward rejects D256, unequal head counts, causal masks and custom
  mask modifiers. MXFP8 backward separately requires D128 and MHA.
- [`flash_bwd_sm100.py`](https://github.com/facebookresearch/ads_model_kernel_library/blob/2889aefba0d03b3eecf779eed9f4097066ecdc5d/lp_fa4/src/lp_fa4/cute/flash_bwd_sm100.py):
  the block-scaled kernel constructor itself asserts D128, GQA ratio 1 and
  noncausal execution. Removing Python interface guards is insufficient.
- [`flash_fwd_sm100.py`](https://github.com/facebookresearch/ads_model_kernel_library/blob/2889aefba0d03b3eecf779eed9f4097066ecdc5d/lp_fa4/src/lp_fa4/cute/flash_fwd_sm100.py):
  the block-scaled constructor requires padded Q/K/V head dimensions of 128
  and disables two-CTA instructions. Existing separate D256 kernel files do
  not establish an available MXFP8 D256 path.

An isolated CPU source audit executes the actual scalar AST guard statements,
without importing Torch, CUTLASS or compiling CUDA. The supported D128 noncausal
MHA control passes those guards. Changing only dimension to 256, only GQA ratio
to 4, or only causality to true each fails the backward constructor guard.
The combined Qwen shape fails, and its D256 source fails the quantizer shape
guard. These are rejection reproductions, not numerical kernel validation.

Receipt: `results/b200_mxfp8_fa4_feasibility/source_compatibility.json`;
executed audit source: `results/b200_mxfp8_fa4_feasibility/executed_source.py`.
The receipt records the upstream revision and SHA-256 hashes of seven inspected
source/manifest files. The current recipe's shape is independently recorded in
`experiments/b200_bf16_fa4/kernel_canary.py` and its completed native receipt.

Repeating each KV head four times could preserve GQA arithmetic through an
MHA kernel, with additional storage, quantization and backward reduction cost;
it would not solve D256 or causal support. Splitting D256 into two independent
D128 attention calls changes softmax and is not a faithful implementation.
Masking the output of noncausal attention cannot restore causal semantics.

## What can be reused and what must change

Reuse `packed_fa4_interface` in `src/gleipnir/packed_sequences.py` as the
integration boundary once a compatible kernel exists. It already delivers
unpadded `(tokens, heads, dimension)` Q/K/V with independent cumulative sequence
offsets. Replace only the eight full-attention layers; keep the 24 FlashQLA
GDN layers, BF16 frozen base and FP32 master LoRA adapters fixed initially.
Attention operand quantization is separate from QLoRA base-weight quantization;
ordinary LoRA still needs gradients through the frozen projections.

Upstream `Mxfp8VarlenMeta` independently pads each example's scale storage to
128 tokens while retaining compact payloads. Preserve this separation so scale
groups cannot couple examples. The explicit forward/backward functions would
need a `torch.autograd.Function` which saves quantized operands, metadata,
output and LSE and returns Q/K/V gradients to the surrounding BF16 operations.
Start with BF16 returned gradients and FP32 dQ accumulation; FP8 internal MMA
does not require exposing FP8 gradients to PEFT. Native MXFP8 gradient outputs
and fused projection producers are later optimizations.

There is also a causal quantization concern to resolve in a kernel port. The
public `_quantize_seq_segment` computes scales over blocks of 32 sequence
positions; forward consumes sequence-oriented quantized V. A future token can
therefore change the scale and rounding of an earlier V even when attention
scores are causally masked. This is an inference from the quantizer, not an
observed Gleipnir failure. Future-token perturbation checks and a causality
policy are necessary alongside attention masking. Packing isolation alone does
not test this property.

The public convenience quantizer separately constructs head and sequence
orientations; do not assume the article's fused producer pipeline or single
transpose-invariant payload is automatically supplied by this API. The
upstream benchmark prequantizes outside kernel timing. A training integration
must count all conversion, scale-layout, storage and gradient costs.

## Proposed screen after compatible kernels become available

Hypothesis: MXFP8 attention reduces complete update time relative to the current
BF16 FA4 recipe, including quantization and backward costs, with usable adapter
gradients. First require causal D256 GQA in both directions, or an explicitly
documented equivalent GQA expansion; stop at unsupported shape or masking.
Use an isolated environment on the existing authorized B200 and persistent
network-volume compiler caches. Pin the candidate source revision and runtime;
do not overwrite the validated environment or reuse its startup receipt.

Run native Qwen-shape forward/dQ/dK/dV comparisons against FP32 causal math and
the current BF16 FA4 kernel, including singleton and ragged lengths near 32- and
128-token boundaries. Record strict errors and future-token/example
perturbation effects separately. Then run fresh whole-model eager/compiled
packing comparisons, memory checks and finite/missing-gradient checks. The
earlier user acceptance of approximately 8% BF16 FA4 gradient difference is
not a new MXFP8 validation receipt or automatic precision acceptance.

If diagnostics support training, reuse the frozen 320-row seed-0 systems cohort,
targets and initial FP32 adapter from the completed FA4 screen. Compare matched
20-update BF16 FA4/MXFP8 trajectories and full physical token partitions, with
ten complete warmup updates and ten complete measured updates; include every
measured batch and reverse the order for replication. Stop on failed numerical
or isolation checks, fallback, version/input drift, OOM, nonfinite values or
missing gradients. Record startup separately from warmed update time.

This systems screen selects no model on a held-out set. Any later quality
selection must use the existing grouped validation split and separately reported
held-out task/source/model-family metrics; never the final test set. Do not
promote from article throughput or training-loss similarity alone. Since only
eight of 32 layers use full attention, whole-training benefit depends on their
measured runtime share, not their layer count.

The immediate next dependency is a causal D256 MXFP8 forward/backward kernel
port or upstream support. No executable training backend was added while those
requirements remain unmet.
