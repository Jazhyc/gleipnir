# SwiGLU overhead and direct FP4 output follow-up

Date: 2026-10-06. The user authorizes two sequential stages: remove overhead
from the existing fused producer, then try native FP4 output. Keep the selected
[attention-FP4 reference](../decisions/b200_attention_fp4_inference_baseline.md)
and its saved controls: 193,224 warm c128 input tokens/s, 156.82 ms warm c1
median and macro/pooled development AUROC 0.878553/0.889729. Use the existing
NC2 B200; retire API/engine 93539/93625 before changing kernels. Keep the pod,
merged weights, master adapters and shared caches. No final-ID selection.

## Stage one: symbolic rows and in-epilogue inverse scaling

Generate an isolated licensed copy of NVIDIA's pinned SwiGLU kernel, retaining
the prior N192 scale-layout correction and inference-only store removal.
One symbolic-M plan accepts all nine native row counts, including M1537.
Native TMA handles partial row tiles without the old allocation/zero/copy
padding sequence. Read the existing activation and frozen-weight inverses
directly inside the epilogue, calculating their reciprocal once per thread's
output row. Preserve raw BF16 GEMM rounding, FP32 descale followed by BF16
rounding, and the activated BF16 boundary before the existing whole-row packer.

Preserve `fp4_swiglu_overhead01`: compilation rejects the unit-dimension stride
of the synthetic FP4 descriptor before GPU arithmetic. Correct the descriptor
to the canonical product stride. `overhead02` passes all nine cases with one
compile but is approximately twice as slow because its source places scaling
division inside the per-value loop. This is a suspected implementation cost,
not a hardware-counter diagnosis. Hoist scale calculation and use the matching
approximate reciprocal in `overhead03`.

`fp4_swiglu_overhead03` passes all nine cases: rows
1/17/129/1536/1537/2304/4096/29184/32768. Zero/extreme rows, isolation and
changed-input CUDA-graph replay pass at the unchanged 1% native ceiling.
Down-projection outputs are exactly equal to the existing producer in all
fixtures. Only one native compile occurs across row counts; no padding is used.
Complete producer speed ratios against FROST gate/up plus fused SiLU/packing:

| Rows | Speed ratio |
| ---: | ---: |
| 1 | 0.8203 |
| 17 | 0.9975 |
| 129 | 0.9994 |
| 1,536 | 1.1132 |
| 1,537 | 1.0230 |
| 2,304 | 1.0891 |
| 4,096 | 1.0617 |
| 29,184 | 1.0561 |
| 32,768 | 1.0413 |

Ratios above one are faster. Five samples of 32 CUDA-graph replays after five
warmups; packing and descale are included, builds and frozen-weight packing
excluded. These are operator gains against the selected producer, not gains
against the earlier fusion trial or serving claims.

`fp4_swiglu_overhead.json` retains the attention-FP4 stack and its tuned plans,
uses this fused producer for M>=1536, and preserves the recorded reference path
for smaller rows. The worker compiles its single native plan before capture;
source-bound admission and live dispatch from all 32 MLPs remain required.
The associated serving screen starts as driver 94653. Its throughput, latency
and AUROC remain pending. Twenty-four focused CPU checks and Ruff pass.

## Stage two: direct packed output

Prepared intervention: emit 16-element E4M3 scales and hardware E2M1 packed
bytes from the native SwiGLU epilogue, removing activated BF16 stores/reads and
the separate pack launch. The first candidate uses a fixed unit global inverse
instead of the reference's whole-row dynamic inverse. This changes quantization
and is explicitly separate from stage one's arithmetic-preserving overhead
work. Bind independent mathematical packing and actual-decoded GEMM references;
retain baseline precision separately from native arithmetic admission. Serving
and development AUROC follow only after supported/finite native validation.
