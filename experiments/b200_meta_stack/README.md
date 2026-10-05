# Meta optimization campaign for causal D256 GQA

Hypothesis: native scheduling and fusing the actual Qwen producers, rather than
only fusing conversion passes, improve complete training updates on the existing
NC2 B200. Baselines are native BF16 FA4 and the direct-varlen/fused-square MXFP8
paths. Keep the same source cohort, initial FP32 rank-128 adapters, logical batch
32, packing, BF16 MLPs and FlashQLA GDN layers. Do not promote on training loss.

Audit every blog technique against the exact pinned source. Existing equivalents
(persistent forward, compact jagged payloads, TMEM scale aliasing, packed scalar
instructions and fixed P scaling) count as already present, with source evidence.
The FP16 global dQ reduction targets a global partial-gradient accumulator that
our separated dQ implementation does not have. Do not add such an accumulator
solely to make the technique applicable. Meta's D128 noncausal MHA pipeline
cannot be substituted for our causal D256 GQA graph.

Interventions: test persistent dQ/dKdV scheduling independently and together;
probe warp-shared online dS maxima and wider dQ stores; fuse head RMSNorm,
partial rotary embeddings and MXFP8 preparation while retaining BF16 rounding
and a training backward; pilot GEMM/LoRA/norm/rotary/quantization producer fusion
on the actual projection geometry. Producer pilots include preparation and
backward costs where supported, and must name forward-only measurements.
Record unsupported upstream fused training interfaces as explicit declines.
Also test fixed forward P scales of 16 and 256 with a zero rescale threshold:
scale only the Tensor Core payload and matching E8M0 metadata, leaving the
probability sums/LSE unchanged. The zero threshold bounds unnormalized P at one
and prevents saturation caused by multiplying the old threshold-four P by 256.
Producer gates require at least 99.9% FP8 code agreement, bit-exact native scales
and at most 1% backward/integration relative L2 versus the separate producer.
The projection pilot uses frozen BF16 weights or a separately prepared MXFP8
copy and FP32 master LoRA updates. It is Q-only, forward-only; gate/K/V and
projection backward are excluded. Reject BF16 pilot code agreement below 99.5%
or MXFP8 agreement below 99.5% against a dequantized-operand GEMM oracle.
Agreement with the original BF16 projection is recorded separately; such a
pilot cannot establish training quality.

Run native mixed-length/singleton/isolation/replay/poison checks first. Each
variant runs in an isolated process so a CUDA assertion/hang cannot contaminate
later candidates. Stop each failed variant on compile failure, nonfinite or
missing gradients, isolation failure, OOM or a 10-minute bounded timeout. Preserve
failed receipts. Compare frozen diagnostic shapes with six warmups and ten timed
repetitions. Only execution-correct candidates with a measured diagnostic gain
advance to at most 20 full-model updates, ten warmup and ten measured. Retain
strict errors and the existing timing-only authorization; require at least 5%
complete-update improvement to recommend a recipe. A whole-model profile is a
separate measurement, excluded from timed updates. No held-out quality selection.

Reuse persistent network-volume compiler caches and pinned runtimes. No new
capacity, environment upgrade, upstream source mutation or default change.
Outputs: `results/b200_meta_stack/`; logs: `logs/runpod/b200_meta_stack/`.
This session has no in-chat heartbeat scheduler; inspect bounded runs during
the active turn and do not promise monitoring after the turn ends.

Source: [Meta blog](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/),
public Meta revision `2889aefba0d03b3eecf779eed9f4097066ecdc5d`, NVIDIA revision
`51d9d06b574222378a3d806009accab098e73705`. Project implementations of ideas are
distinguished from the unavailable public producer epilogues and from upstream
MIT/Apache licensed source (see the earlier experimental notices).
