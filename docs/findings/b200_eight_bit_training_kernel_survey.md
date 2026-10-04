# Eight-bit attention training kernel survey

Date: 2026-10-05. Online survey and inspection of public upstream source.
No GPU execution, dependency changes, or new training runs were performed.

## Finding

NVIDIA cuDNN Frontend's development FROST engines provide a plausible MXFP8
forward/backward path for our eight causal D256 GQA full-attention layers on
B200. The backward adapter currently rejects packed variable-length inputs,
so this is an integration candidate rather than a replacement for the validated
BF16 FA4 recipe. No public eight-bit forward/backward implementation compatible
with our 24 Qwen3.5 Gated DeltaNet (GDN) layers was identified. Keep BF16 FA4 and
FlashQLA as the standard pending a separate matched training screen.

This survey distinguishes attention-core arithmetic, recurrent-state storage,
and surrounding linear projections. FP8 weight exports or inference KV caches
do not establish low-precision training of either attention operator.

## Candidates

| Candidate | Public evidence | Fit to our training recipe |
| --- | --- | --- |
| cuDNN Frontend FROST MXFP8 | Blackwell D256 causal GQA forward and backward | Strongest full-attention candidate; backward is dense BSHD only |
| SageBwd | INT8 training research, six of seven attention matmuls quantized | Could not locate released backward code in the official repository or its listed branches; D256 packed compatibility unverified |
| Delta-Matching | Block-scaled FP8 attention training; D256 and hybrid GDN/GQA experiments | Paper promises a future implementation release; recurrent layers explicitly remain BF16 |
| LeapQuant | Eight-bit GDN/KDA recurrent-state quantization, including B200 | Inference method with buffered high-precision updates; no training backward established |
| FlashQLA / FLA GDN | Existing differentiable GDN kernels | FlashQLA requires BF16/FP16 QKV; no supported quantized GDN training path found in inspected FLA sources |
| Transformer Engine linear attention | cuDNN-backed experimental GDN modules | QKV must be BF16/FP16; modules explicitly ignore FP8 autocast |

Primary paper sources: [SageBwd](https://arxiv.org/abs/2603.02170),
[Delta-Matching](https://arxiv.org/html/2609.37852v1), and
[LeapQuant](https://arxiv.org/abs/2609.38166). Both September papers were submitted
on 2026-09-29. Delta-Matching Appendix B.2.2 explicitly separates FP8 projections
from BF16 recurrence; its hybrid-model results do not establish FP8 GDN kernels.
SageBwd's paper is evidence of a training method, not verified available code.

## NVIDIA path and limits

Inspected cuDNN Frontend development revision:
`51d9d06b574222378a3d806009accab098e73705`.
The [merged backward PR](https://github.com/NVIDIA/cudnn-frontend/pull/904)
and the actual
[backward engine capabilities](https://github.com/NVIDIA/cudnn-frontend/blob/51d9d06b574222378a3d806009accab098e73705/python/cudnn/sdpa/bwd/engines.py)
provide exact D256, E4M3 payloads, causal masking, and GQA on SM100 Blackwell.
Returned gradients use BF16/FP16. Higher-precision reductions and accumulation
remain; eight-bit matmul operands do not mean every operation uses eight bits.
The corresponding
[forward engine](https://github.com/NVIDIA/cudnn-frontend/blob/51d9d06b574222378a3d806009accab098e73705/python/cudnn/sdpa/fwd/engines.py)
also serves D256 causal GQA and supports packed layouts.

The backward adapter accepts physical BSHD layout and fixed per-batch sequence
lengths. It rejects THD, sequence-length/padding masks, sliding windows and bias.
Its masked tile tails permit arbitrary dense sequence lengths; they do not
provide isolation between concatenated examples. Underlying backward kernels
contain variable-length machinery, but the public adapter does not expose a
validated packed path. Forward packed support alone is insufficient for training.

The [backward adapter](https://github.com/NVIDIA/cudnn-frontend/blob/51d9d06b574222378a3d806009accab098e73705/python/cudnn/sdpa/bwd/api_dsl_mxfp8_sm100.py)
needs separately quantized transposed operands and their scale buffers, plus
high-precision output, output gradient and LSE. It repacks scale layouts before
launching dQ and fused dK/dV kernels. Include quantization, repacking, launch
overhead, saved tensors and backward in measured update time. This is a
development-source path, not proof that the currently installed cuDNN or TE
selects it. FROST engines require explicit opt-in and verified engine selection.

Transformer Engine's general FP8 attention support matrix is not sufficient:
the inspected fused-attention selector rejects Blackwell FP8 THD heads above
128. Prototype the direct cuDNN Frontend path in an isolated environment and
pin dependencies without disrupting FlashQLA import precedence or shared caches.

## Bounded next screen

The first faithful full-attention prototype can call dense causal MXFP8
separately for each example, preserving sequence isolation. This will sacrifice
packed batching and may lose throughput; measure the complete update against
BF16 FA4 before investing in a packed backward adapter. Do not reinterpret the
concatenated packed stream as one causal sequence. A production integration
needs sequence-local payload/scales and forward/backward variable-length tests.

Keep FlashQLA recurrence in BF16. A separate practical eight-bit experiment is
FP8 GEMMs for surrounding frozen base projections using
[torchao float8 training](https://docs.pytorch.org/ao/stable/api_reference/api_ref_float8.html).
That API replaces linear layers, not GDN recurrence. LoRA still needs gradients
through frozen projections; preserve FP32 master adapters and verify PEFT,
checkpoint and input-gradient compatibility. There is no measured speed or
quality conclusion for this configuration.

New precision requires fresh numerical, sequence-isolation and finite-gradient
checks under the existing experiment policy. The prior user acceptance of the
BF16 FA4 gradient difference is not validation of these kernels. A longer
matched trajectory matters: Delta-Matching reports that short-run loss agreement
can conceal FP8 backward inconsistencies.

## Source snapshots

| Upstream repository | Inspected revision | Relevant source |
| --- | --- | --- |
| NVIDIA/cudnn-frontend | `51d9d06b574222378a3d806009accab098e73705` | `python/cudnn/sdpa/{fwd,bwd}/engines.py`, `bwd/api_dsl_mxfp8_sm100.py` |
| NVIDIA/TransformerEngine | `d0b4b32c8e9046246a7ea20025eb06eaba3d7a6b` | `common/fused_attn/fused_attn.cpp`, `pytorch/attention/linear_attention/base.py` |
| QwenLM/FlashQLA | `da06429d54b0f577de0a638f451ac8f0b395e0ac` | `flash_qla/ops/gated_delta_rule/chunk/__init__.py` |
| fla-org/flash-linear-attention | `8024667ab58fdd8986587147fca71fecc017f977` | `fla/ops/gated_delta_rule`, `fla/ops/common` |
| thu-ml/SageAttention | `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5` | README, public package and remote branch listing; no located SageBwd implementation |
| hao-ai-lab/flash-attention-fp4 | `c206aa7e34e93a06c73d94c710ed87cde13f21f0` | Low-precision forward kernels and autograd interface; no verified D256 eight-bit backward path |

The current FlashQLA upstream revision matches our existing pin. Its
[dtype guard](https://github.com/QwenLM/FlashQLA/blob/da06429d54b0f577de0a638f451ac8f0b395e0ac/flash_qla/ops/gated_delta_rule/chunk/__init__.py)
explicitly permits only BF16/FP16. Transformer Engine's
[linear-attention base](https://github.com/NVIDIA/TransformerEngine/blob/d0b4b32c8e9046246a7ea20025eb06eaba3d7a6b/transformer_engine/pytorch/attention/linear_attention/base.py)
explicitly rejects FP8 QKV and bypasses FP8 autocast.

Official FlashAttention-3's documented FP8 path is forward-only. SageAttention's
available inference kernels and HAO's low-precision forward paths do not establish
a usable D256 packed eight-bit training implementation. The earlier
[Meta LP-FA4 assessment](b200_mxfp8_fa4_feasibility.md) remains applicable to that
specific release; NVIDIA's separate engine does not remove its source guards.
