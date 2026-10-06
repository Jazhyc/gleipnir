# Combined preparation and tuned FP4 GEMMs as the B200 inference reference

Decision date: 2026-10-06. The user explicitly says, “I think we can use this
as a reference,” accepting that small gains can accumulate. Select the completed
`fp4_gemm_tuned_serving01` stack for subsequent inference optimization. This
supersedes the [MXFP8-only selection](b200_mxfp8_inference_baseline.md).

Use native FROST FP4 MLP and large GDN projections, combined vendor packing /
fused normalization / fused SwiGLU preparation, shape-dependent FP4 output
projection tiles and adapted cuDNN MXFP8 attention prefill. Use N128 tiles for
MLP down and GDN output at M<=4096 and the original N256 tile otherwise.
Preserve BF16 small gates, recurrence, convolution, cache/decode and FP32
gates/state, the FP32 master and ephemeral merged model. Capacity and scoring
remain unchanged.

## Evidence and acceptance

The five-pass warm c128 median is **182,682 input tokens/s**, −1.66% versus
the preceding combined-preparation experiment and +5.03% versus the former
selected MXFP8 reference. A separate fully warm c1 confirmation gives
**147.00 ms median / 270.26 ms p95**. Its initial post-startup c1 median was
163.41 ms; warmup history limits causal attribution of the latency difference.
All 160 executable native tile cases agree exactly with the original BF16
outputs and pass isolation/changed-input replay checks. Preserve the four
upstream TMEM compile rejections.

Source-macro/pooled AUROC is 0.884608/0.908150 on the frozen training-seen
systems-dev cohort, −0.0584/+0.0137 percentage points against saved combined
full-sweep scores. Preserve all ranking, calibration, threshold and repeat
diagnostics. These results establish no final-ID or broader production-quality
claim. See the [complete finding](../findings/b200_fp4_gemm_tuning.md).

The new `quality_acceptance.json` records the user's selection separately as
`user_accepted_finite`. Strict master-score, MXFP8 FP32-reference and vendor
decoded-precision checks retain their failed status; new tile checks pass.
No tolerance is widened, and matching inputs, finite scores, source bindings
and nonzero adapter effect remain required for later kernel changes.

## Bound controls and worker reuse

`experiments/b200_inference_benchmark/baseline.json` binds the result summary,
executed recipe copy, workload, full sweep, warm throughput, warm latency,
native GEMM validation and explicit acceptance by checksum. Archive the preceding
selection under `baselines/frost_fp4_mlp_gdn_cudnn_mxfp8_prefill.json`.
The original executed configuration retains its historical combined reference;
the live configuration uses `high_reference: selected` for future comparisons.
The runner now defaults to `fp4_gemm_tuned.json`.

Use the five-pass warm confirmation for primary throughput comparisons and
the separate c1 confirmation for warm latency comparisons. Preserve the original
full-sweep timings and score controls. Selection and a client-only reference
change do not change active kernels or capacity, so no restart or control replay
is required. API/engine **89879/89962** stay resident on the existing NC2 B200,
port 8010. Stop that server before changing active kernels/settings or starting
another GPU worker. No new capacity is created or terminated.

## Remaining measured cost

The latest kernel profile precedes tile tuning. GEMMs are the largest combined
category: FP4 24.03% plus BF16 7.97%, about **32% of summed GPU kernel time**.
Gate/up and GDN input account for 73.2% of FP4 GEMM time. Normalization/gates/
layout is 19.64%, GDN core 14.37%, fused FP4 producers 13.00% and MXFP8 attention
core 9.65%. Large-row tile choices are unchanged, making this a useful starting
estimate rather than a fresh profile. GPU kernel shares do not establish
Tensor Core saturation or identify compute versus memory stalls; those counters
remain unmeasured.


Later on 2026-10-06, the user-authorized
[backend comparison](../findings/b200_fp4_backend_comparison.md) retains FROST.
After stopping the old server for native work, restore the same recipe as
API/engine **91253/91317**, port 8010. Existing quality and performance controls
remain bound; four warmup requests and current-PID runtime audits verify the
restored worker. The selection is unchanged.
