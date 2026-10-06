# Matched FP4 GEMM backend comparison on B200

Date: 2026-10-06. The user requests a comparison of FP4 GEMM backends after
selecting the combined-preparation, shape-tuned FROST serving reference.
**Keep FROST:** none of the four alternatives improves the complete operator
under the declared shape/row-band selection rule. Restore and retain the
unchanged serving recipe; no alternative is installed into the model.

## Complete-operator native timings

At M32768, milliseconds per projection, including the required descaling:

| Projection, K × N | FROST | cuDNN | CUTLASS | TRT-LLM | CuTe-DSL |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP gate/up, 2560 × 18432 | 0.6124 | 1.0327 | 1.0267 | 1.2917 | 1.0659 |
| GDN input, 2560 × 12288 | 0.4044 | 0.6713 | 0.6659 | 0.8286 | 0.6852 |
| MLP down, 9216 × 2560 | 0.2574 | 0.3126 | 0.3080 | 0.4381 | 0.3528 |
| GDN output, 4096 × 2560 | 0.1262 | 0.1821 | 0.1785 | 0.2314 | 0.1824 |

CuTe is measured in the corrected follow-up against its own matched FROST
control; those four controls are 0.6124/0.4069/0.2573/0.1262 ms. Keep the
two runs separate. Native timings are not HTTP throughput measurements.

Small-band geometric-mean speed ratios versus FROST, over M129/1536/2304/4096:

| Projection | cuDNN | CUTLASS | TRT-LLM | CuTe-DSL |
| --- | ---: | ---: | ---: | ---: |
| MLP gate/up | 0.714 | 0.722 | 0.605 | 0.714 |
| GDN input | 0.795 | 0.747 | 0.656 | 0.712 |
| MLP down | 0.828 | 0.928 | 0.716 | 0.832 |
| GDN output | 0.866 | 0.862 | 0.751 | 0.928 |

Higher ratios are better. A few tiny-row alternatives beat individual FROST
cases, but none wins the prescribed bands. Large-band ratios over M29184/32768
are 0.600/0.604/0.818/0.692 for cuDNN, 0.602/0.615/0.838/0.706 for CUTLASS,
0.477/0.489/0.588/0.548 for TRT-LLM and 0.575/0.588/0.730/0.691 for CuTe.
This bounded comparison does not establish a global backend/tactic optimum.

## Matched arithmetic and execution

Use all four fixed projection shapes and rows
1/17/129/1536/2304/4096/29184/32768. Preserve identical packed FP4 payloads,
128x4 E4M3 block scales, scalar weight inverse and per-row activation inverses.
TRT-LLM weight payloads and scale bytes are permuted once without requantization;
activation packing is unchanged and excluded for every backend. Preserve raw
BF16 GEMM rounding, then FP32 row descaling, then final BF16 rounding. Alternative
backends run raw GEMM with scalar alpha one and the existing descaling kernel.
FROST retains its fused row-descaling epilogue. Different tactics may change
the raw GEMM implementation, but all executable cases observe zero numerical
relative-L2 difference, exact zero rows, unchanged other rows and matched
changed-input graph replay at the unchanged 1% admission ceiling.

The initial persistent worker completes in 253.1 seconds: 128 successful cases
and 32 CuTe API failures. Preserve these failures. FlashInfer 0.6.12 references
`cute.make_fragment`, absent in the pinned CUTLASS DSL 4.8.0. The documented
[register-tensor API rename](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/changelog.html)
permits explicit process-local aliases to `make_rmem_tensor` and its `_like`
variant. Do not change installed packages or precision. A targeted second worker
compares only corrected CuTe and FROST, completing 64 successful cases in
49.8 seconds. All four alternatives therefore pass their 32 arithmetic cases;
including both FROST controls, 192 cases pass and the 32 historical failures
remain recorded. CuTe compatibility is never installed into the serving worker.

Autotune uses the exact eight row buckets, retaining 152 cached configurations
at `.cache/flashinfer/autotune/gleipnir_fp4_backends.json`. Bound CuTe to at most
eight representative supported tactics per shape/row, recording the original
valid count and selected candidates. Tuning, JIT builds and one-time weight
permutations are excluded from timing. Report medians of five samples of 32
CUDA-graph replays after five graph warmups. Use an independent seeded CUDA
generator so autotuning does not change benchmark operands. Shared compilation
and kernel caches persist on the network volume.

## Interpretation and retained reference

The generic interfaces require a separate BF16 row-descaling pass, adding a
read and write of the output. For the M32768 gate/up output, that is another
2.416 GB of logical tensor traffic. The current FROST implementation already
fuses this work. This is an integration cost that raw-GEMM-only comparisons
would miss; the study does not separately time its contribution or identify
Tensor Core/DRAM/L2 stalls. A different backend alone is not a measured gain
for this contract. Deeper GEMM/SwiGLU fusion remains a separate candidate.

No new model kernel is selected, so reuse the reference's archived warmed
throughput, latency and AUROC rather than replaying an unchanged control or
claiming a new quality pass. Keep 182,682 input tokens/s and 147.00 ms warm
median latency as the comparison anchors. Strict failed precision receipts
retain their classification. This comparison contains no final-ID evaluation.

Stop selected API/engine 89879/89962 before native work. The unchanged reference
is restored as API **91253**, engine **91317**, port **8010**, on the same NC2
B200. Four bounded warmup requests advance with finite decision scores and
matching token counts; live attention/preparation/GEMM usage audits pass for
the new worker PID. Reuse the checksum-bound prior HTTP quality receipt and
record that reuse explicitly. No full control replay or new capacity is launched.
`campaign.json` records completion, both native results and the restored worker.

Artifacts: `fp4_backend_compare01.json`, `fp4_backend_compare02.json`, their
exact source archives, `fp4_backend_restore01` and `fp4_backend_collection01`,
under `results/b200_attention_gdn_serving/`. The collection includes original
failures, installed FlashInfer implementation copies, autotune cache and logs.
