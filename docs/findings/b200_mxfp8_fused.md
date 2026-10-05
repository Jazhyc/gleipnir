# Fused MXFP8 preparation and square block scaling

2026-10-05. The user requested adopting compatible optimizations from the
[Meta low-precision FA4 report](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/)
and separately authorized trying project-specific optimizations. This experiment
retains NVIDIA's causal D256 GQA kernels and implements fused operand/scale
preparation, optional transpose-invariant 32x32 scaling, and fused device
boundary checks. The existing BF16 FA4 default remains unchanged.

## Scope and implementation

`nvidia_mxfp8_fused` uses the existing 32x1 / 1x32 arithmetic, producing both
payload orientations and native consumer scales in one Triton pass per operand.
`nvidia_mxfp8_square` uses one 32x32 maximum and one payload for both orientations.
Both retain compact THD payloads, independent sequence-local tile origins and
native forward/backward scale layouts. Four preparation launches replace eight
quantizer and eleven repack launches; boundary checks use one fused predicate
kernel plus the existing asynchronous CUDA assertion. All validation predicates
and the declared 32-example / 131,072-token host envelope remain enforced.

These apply Meta's producer/layout fusion and transpose-invariant scaling ideas
at the attention boundary. BF16 tensors still enter the producer and BF16 input
gradients leave attention. This is not a port of Meta's unpublished RMSNorm/GEMM
epilogues, FP8 projection GEMMs or FP8 returned gradients. Qwen's Q/K head
normalization and rotary embeddings precede the attention boundary; direct
projection-epilogue quantization would need to account for those operations and
FP32 LoRA masters. Meta's FP16 global dQ reduction is not directly applicable:
the pinned separate NVIDIA dQ kernel retains the accumulated result in TMEM and
stores the final gradient in its epilogue. Native backward kernels and online
dS scaling remain unchanged.

The public Meta revision remains
`2889aefba0d03b3eecf779eed9f4097066ecdc5d`; its attention path still lacks causal
D256 GQA support. The fused producer is project code using NVIDIA consumer
layouts at revision `51d9d06b574222378a3d806009accab098e73705`, not copied
unpublished Meta code. Third-party notices link the retained NVIDIA licenses.

## Native execution checks

Fresh `dual02` and `square02` execute on the existing US-NC-2 B200 with shared
persistent caches. Mixed lengths are
`[1,3,31,33,63,65,127,129,255,257]` with 16 and 4 heads. Structured operands
vary in magnitude across tokens and feature blocks, exercising nonuniform
scales. Every payload, native row SFA/SFB and column SFB layout, and live compact
forward scale tile matches its oracle byte-for-byte. Dual mode uses the original
packed NVIDIA producer/repacker as oracle; square mode uses an independent
tensor implementation of 32x32 maxima and canonical scale packing.

Dual output and all three input gradients match the old packed MXFP8 path
exactly. Both modes pass finite gradients, singleton semantics, zero measured
cross-example leakage, changed-cut CUDA graph replay and poisoned storage.
Independent FP32 parity still fails:

| Mode | Forward relative L2 | dQ | dK | dV | Strict 2% / 5% |
| --- | ---: | ---: | ---: | ---: | --- |
| Fused dual | 4.2550% | 6.6065% | 6.5999% | 4.6236% | Failed |
| Fused square | 4.2550% | 6.6065% | 6.5999% | 4.6236% | Failed |

Near-equal errors on this random attention probe do not establish identical
square/dual arithmetic on real model activations. Their producer comparison
uses a separate structured-magnitude input and validates square scaling against
its own mathematical reference.

`boundaries01.json` separately checks eight valid/invalid predicate cases,
including 32 examples, empty/decreasing offsets, incorrect first/final offsets
and exceeded maximum length. All predicates match expectation. Future-token
perturbations across a 32-token quantization boundary leave the earlier payload
unchanged. Within a block, dual column payload changes; square row and column
payloads both change. This is within-block scale coupling despite a causal
attention mask, not exact prefix-invariant arithmetic. Retain that limitation
and separate numerical failures; packing isolation does not establish strict
causality of quantized operands.

Receipts:

- `dual02/kernel_canary.json` SHA256:
  `fb5be01de46a744c6ad7a50f5fc41fe6e17279876324ade1aa936418af5c018c`.
- `square02/kernel_canary.json` SHA256:
  `39d11ec07c8c435addd7ce41820425d9b31a81bc09011f7aea2aa8fb6659aec3`.
- Independent helper `square_reference.py` SHA256:
  `2686b9e393825dbfafd9c945124a98438417a1a65c95b7594801c46132abad56`.

Earlier `dual01` / `square01` remain intact. The second attempts strengthen
the producer reference and include the subsequently fused boundary checks.

## Matched attention diagnostic

`benchmark02` measures four controlled synthetic shapes in one process using
Torch 2.11.0+cu130, pinned FA4 and NVIDIA overlays, the same B200 and persistent
caches. Six warmups precede ten uncaptured repetitions per condition; twenty
graph replays separately remove most host dispatch. Order reverses on alternate
shapes. All conversion, native backward and scratch work are included.

| Controlled lengths | BF16 FA4 wall ms | Old MXFP8 | Fused dual | Fused square |
| --- | ---: | ---: | ---: | ---: |
| 301,280,250,230,201 | 1.063 | 6.018 | 4.586 | 4.154 |
| 4 x 4,096 | 2.417 | 5.876 | 4.518 | 4.574 |
| 14,373,1,000,992 | 6.438 | 6.061 | 5.850 | 5.117 |
| 24,521 | 18.253 | 13.025 | 12.924 | 12.456 |

| Controlled lengths | BF16 FA4 graph ms | Old MXFP8 | Fused dual | Fused square |
| --- | ---: | ---: | ---: | ---: |
| Short pack | 0.102 | 0.289 | 0.259 | 0.198 |
| Balanced pack | 2.303 | 2.488 | 2.575 | 2.229 |
| Skewed pack | 6.350 | 5.198 | 5.691 | 4.916 |
| Long singleton | 18.135 | 12.587 | 12.717 | 12.237 |

Actual GPU kernel count falls from 38–39 to 16 per call. Square scaling cuts
some operand traffic and reductions; dual fusion can increase GPU preparation
cost on balanced/skewed packs despite fewer launches. Remaining uncaptured
Python/FFI/allocation overhead is substantial, especially on shorter packs.
Long-input total attention time improves about 31.8% versus FA4 but only 4.4%
versus old MXFP8 in this diagnostic. Do not infer complete-update performance
or exact GDN/MLP time shares from these attention-only results.

`benchmark01` records the preceding implementation before boundary fusion and
is preserved. Profiler attribution counts only actual Chrome kernel events,
excluding synthetic GPU annotations. Timings exclude profiler overhead;
clock/order variation remains visible and no long repeated campaign is claimed.

## Whole-model screen

The checksum-bound `training_square01` screen completed exactly 20 updates
from the same initial FP32 adapter, 320-row cohort, seed and adaptive packing as
all three historical controls. Ten warmup updates precede ten measured updates;
all physical row contracts match exactly. The measured window contains
1,314,331 tokens and 73 physical rows. No fresh control replication is claimed.

| Complete-update recipe | Mean seconds/update | Peak allocated GiB |
| --- | ---: | ---: |
| Historical BF16 FA4 | 4.08648 | 145.242 |
| Historical dense MXFP8 | 5.29781 | 146.852 |
| Historical direct-varlen MXFP8 | 3.93793 | 145.098 |
| Fused square MXFP8 | 4.15062 | 144.242 |

Square fusion takes 21.65% less time than dense MXFP8, but 1.57% more than FA4
and 5.40% more than direct-varlen MXFP8. It does not meet the predeclared 5%
complete-update improvement rule. The ten measured updates total 41.5062 seconds;
all twenty update timers total 84.9899 seconds, the trainer loop takes 89.2456
seconds, and the full invocation including fresh gates takes 458.826 seconds.
Attention-only improvements therefore do not establish a training speedup.

Fresh eager/compiled packing gradient relative L2 errors are 19.49%/18.22%,
versus 18.71%/16.66% for direct-varlen MXFP8. Both exceed strict 5% and separate
10% learning ceilings; both runs retain failed receipts. The adaptive gradient
canary is finite but fails strict parity at 36.77%. Cross-example perturbation
and input-gradient leakage are zero in all fresh packing cases. The longest-row
preflight passes with unchanged master adapters and finite gradients. The
explicit earlier timing-only authority permits this bounded comparison and does
not establish learning quality or numerical equivalence.

All twenty updates enforce finite loss and reject missing/nonfinite gradients.
The collected checkpoint contains 256 finite FP32 master adapter tensors; local
checksums match the remote adapter and metadata, and all 24 archived source files
match their recorded hashes. Its final master hash
is `8a1666ad2aecd366e9fd9dca838a9fa32d9c2e54cd815e16e0fc2f04ab06d689`,
different from the shared initial adapter and the earlier dense/direct-varlen
trajectory. The saved adapter file SHA256 is
`2e5b88f7e46a2cff42439c3b201ded32306cca4f5cc813f99305b7518ca36f33`.
Metadata SHA256:
`60fc8b01e01b96a98ec9f9b0d72511b13af2eae8a3bbd8c9bdc484ac63f4b2b1`.
The result, executed-source archive and runtime logs are collected locally under
`results/b200_mxfp8_fused/training_square01/` and
`logs/runpod/b200_mxfp8_fused/`.

Keep BF16 FA4/FlashQLA as the standard. Both fused modes remain explicit
experimental backends. Fused dual has byte-exact native parity with the old
producer; only square mode has a complete-model timing screen here. No held-out
quality validation, fresh FA4 replication or claim of a precise whole-model
bottleneck follows. Short-pack dispatch overhead, slower fused preparation on
some shapes and the native dQ backward kernel remain measured optimization
targets; GDN/MLP shares still require a whole-model profile. Meta's public recipe
also changes tensor-production and native attention internals that this port
has not implemented.

Validation: 68 focused CPU tests pass, including new explicit-precision
routing, bounded authorization, failed-native rejection, square scale-group
scope, and the affected existing packing contracts. Ruff and diff checks pass.
Feature commit: `4910f6b` on `research/b200-mxfp8-fused`.
