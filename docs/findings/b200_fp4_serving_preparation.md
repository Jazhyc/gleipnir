# FP4 activation preparation on the MXFP8 serving reference

Date: 2026-10-06. The user authorizes all three proposed preparation changes:
installed CUDA per-token NVFP4 packing, fused SwiGLU packing, and fused
post-attention residual/RMSNorm packing. The vendor trial finishes with a
five-pass median of **181,786 input tokens/s**, **+4.51%** over the accepted
MXFP8 reference's 173,938. Its score canary fails the existing agreement limits;
retain this as diagnostic evidence, without changing the selected reference.

## Frozen scope and evidence

Use the existing NC2 B200 pod, ephemeral merged BF16 model, all 64 FP4 MLP and
48 large FP4 GDN projections, adapted cuDNN MXFP8 full-attention prefill,
BF16 cache/decode/recurrence and FP32 gates/state. Preserve the FP32 master.
Keep 90% GPU memory, 128 sequence slots, 32,768 scheduled/context tokens,
disabled prefix caching and the constrained one-token scoring contract.
Retire each server before changing kernels. No concurrent control replay or
new billable capacity is required.

The frozen 320 training-seen systems-development rows contain 1,310,581 input
tokens; manifest SHA256 is
`b02af232d76935f2cda2a52213ebce27ae04ca5aaf44898cfe496104adad4266`.
Quality compares saved c128 repeat-median scores from
`fp4_gdn_cudnn_mxfp8_02`. Speed compares the five-pass median in
`mxfp8_confirmation02`. These score and speed references are separate artifacts;
batch-dependent score ties mean quality can differ slightly from that earlier
five-pass quality summary. No result measures final ID generalization.

## Native admission

`fp4_prepare_vendor05` tests K2560/4096/9216 and M1/17/129/32768, including zero
and extreme rows, row isolation and changed-input graph replay. Installed
FlashInfer 0.6.12 emits invalid scales for all-zero rows, so explicitly initialize
their mathematical zero payload and scales, as well as scale padding. Convert
per-token scales to the inverse convention required by FROST. A host scalar
avoids the wrapper's GPU `.item()` path. Do not fall back to another quantizer.

All twelve actual-operand GEMM arithmetic/finite checks pass, with maximum
relative-L2 1.08e-5. The unchanged 1% baseline decoded-precision ceiling still
fails at 1.81%; `passed=false` and `arithmetic_passed=true` remain distinct.
Long-row packing times are 0.0926→0.0462, 0.1354→0.0619 and 0.2892→0.1294 ms
at the three widths. Preserve the earlier missing-PATH, padding-dtype,
zero-row-nonfinite and strict precision failures.

`fp4_fusion_canary01_silu` preserves the failed eager-style BF16 intermediate
rounding attempt. Generated Inductor code instead retains FP32 SiLU and
multiplication until a single BF16 output. Canary02 follows that compiled
contract. Both fusions pass eight cases, native actual-operand GEMM checks,
isolation and changed-input graph replay. SwiGLU decoded values/payloads/scales
are exact in the tested cases. Normalization preserves the BF16 residual exactly;
maximum decoded relative-L2 is 0.000677 and native GEMM error 0.0000923.

At M32768, select eight warps for SwiGLU: producer plus packing falls from
0.806 to 0.457 ms, 1.76 times faster. Select four warps for normalization:
0.499 to 0.110 ms, 4.55 times faster. Native timings use CUDA graphs and include
allocations/layout/inverse work, but exclude HTTP and host dispatch. Each
fusion bypasses the existing opaque pack-and-GEMM boundary with explicit
packed tensors and registered custom operations/fake implementations.

## Vendor end-to-end result

`fp4_prepare_vendor_serving01` completes six quick latency passes and eight
full-cohort throughput passes. Five additional warmed c128 passes in
`fp4_prepare_vendor_confirmation01` reuse the same API/engine PIDs 86965/87130.
Their input rates are 180,718 / 182,389 / 180,956 / 182,640 / 181,786 tokens/s;
exclude the preceding full-cohort warmup. The c1 quick median latency is
150.77 ms versus 149.50 ms for the saved reference, so this change does not
improve interactive median latency. Keep sweep variability separately.

The twenty-row score canary's baseline mean difference is 0.021804 and
correlation 0.993867; baseline-relative and strict master acceptance both fail.
Finite diagnostic continuation does not erase either failure. Full-cohort
source-macro AUROC is 0.900666→0.890636, **−1.00 percentage point**; pooled
AUROC is 0.900238→0.904927, **+0.47 points**. Mean/max score differences are
0.046344/0.442974, with twenty threshold flips. Pooled Brier is
0.137661→0.133417 and FPR at 0.5 is 0.179012→0.160494. Preserve per-source,
partial-AUROC and score-tie diagnostics; small source groups make macro changes
coarse. The worker is retired before the independent SwiGLU trial starts.

## Independent SwiGLU model result

`fp4_prepare_silu_serving01` completes all fourteen passes and passes the
baseline-relative score canary: mean difference 0.014455, correlation 0.998019.
Strict master parity remains failed. Five warm passes in
`fp4_prepare_silu_confirmation01` give median **180,903 input tokens/s**,
**+4.00%**, with rates 179,499 / 181,522 / 181,014 / 180,903 / 179,092.
API/engine PIDs 87814/87891 are reused and subsequently retired.

Source-macro AUROC is 0.900666→0.882396, **−1.83 percentage points**; pooled
AUROC is 0.900238→0.896331, **−0.39 points**. Mean/max score differences are
0.037631/0.330534, with thirteen threshold flips. Native bitwise agreement in
random/extreme fixtures does not establish whole-model score identity.
Inspecting cached serving kernels confirms FP32 `gate / (1 + exp(-gate))`
and multiplication followed by BF16 storage, matching the canary's formula;
the cause of the model-level drift is not established. Do not promote a claim
of exact model equivalence from these fixtures.

## Cache and artifact handling

The previous server's FlashInfer cache lived under `/root/.cache/flashinfer`.
Copy its 28 MB into the network-volume `.cache/flashinfer` directory and set
`FLASHINFER_WORKSPACE_BASE` to the repository root. Native builds refresh after
path relocation and then share this persistent namespace. Preserve existing
Inductor, vLLM, Triton, CuTe and cuDNN cache locations. Startup imports, builds,
compilation and graph capture are excluded from warmed speed measurements.

Native receipts and executed source archives are collected locally and their
source hashes verified. Serving predictions, score failures, source archives
and warm confirmation receipts are also collected. Outputs remain under
`results/b200_attention_gdn_serving/`, logs under the matching Runpod tree.
Focused tests and Ruff pass. The selected optimization baseline is unchanged.
