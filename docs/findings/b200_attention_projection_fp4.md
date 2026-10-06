# FP4 full-attention projections on B200

Date: 2026-10-06. The user authorizes changing the remaining full-attention
projection GEMMs to FP4. The named recipe `attention_fp4.json` extends the
selected tuned FROST reference; experimental SwiGLU fusion is excluded.
The implementation is complete and the warm throughput gain is reproducible,
but interactive latency and monitoring quality worsen. At completion the
recipe remained named pending explicit acceptance. The user subsequently
selects it as the default/reference with that tradeoff recorded; see below.

## Intervention and native validation

Quantize the eight decoder full-attention layers' QKV and output projections:
16 modules total, in layers 3/7/11/15/19/23/27/31. QKV is K2560→N10240,
including its existing query gate; output is K4096→N2560. Apply the established
FROST NVFP4 weight format and vendor rowwise activation packing, retaining raw
BF16 GEMM rounding, FP32 row descaling and BF16 output rounding. Keep QK norm,
RoPE, nonlinear gating, causal cuDNN MXFP8 prefill, BF16 KV/decode/recurrence,
FP32 state/gates, small GDN gates and vocabulary head at their original scope.
Pack only the ephemeral master-derived merged model's serving copies.

QKV uses the existing symbolic-M 128×256 two-CTA FROST tile. Output reuses
the selected GDN-output shape's small/large tiles, since K/N are identical.
Runtime audits identify each actual attention layer by its packed weight
pointer inside the existing opaque linear operator. This prevents scope logging
from entering the compiled model graph. Require all 16 modules, correct
geometry, the current worker PID and the source-bound numerical receipt.
Combined preparation now requires 64 vendor calls rather than 48, plus the
same 32 normalization and 32 SiLU producers. Historical conditions retain
their original 48-call requirement.

`attention_fp4_canary02` passes all 16 cases: both shapes at rows
1/17/129/1536/2304/4096/29184/32768. Require finite outputs, exact zero rows,
unchanged other rows, nonzero changed-row effect and changed-input graph replay.
Compare the first 16 output columns against an independently decoded FP4
FP32 reference, with the same BF16 rounding/descale boundaries and unchanged
1% admission ceiling. Record BF16 quantization error separately; it is not
the native implementation-error tolerance.

| Projection | Rows | FP4 including packing/descale, ms | BF16 GEMM, ms | Speed ratio |
| --- | ---: | ---: | ---: | ---: |
| QKV | 1 | 0.016678 | 0.012572 | 0.754 |
| QKV | 129 | 0.016780 | 0.012712 | 0.758 |
| QKV | 4,096 | 0.057666 | 0.137240 | 2.380 |
| QKV | 32,768 | 0.434652 | 1.066896 | 2.455 |
| Output | 1 | 0.015850 | 0.010412 | 0.657 |
| Output | 129 | 0.014648 | 0.008540 | 0.583 |
| Output | 4,096 | 0.033178 | 0.056926 | 1.716 |
| Output | 32,768 | 0.184866 | 0.433262 | 2.344 |

Ratios greater than one are faster. Use medians of five samples of 16 CUDA-graph
replays after five warmups. Exclude builds and frozen weight packing from both
timings; include activation packing for FP4. TF32 is disabled for the FP32
quantized reference. Native arithmetic errors are roughly 1e-5 relative L2 in
the two largest cases; differences versus BF16 are approximately 14.6%.
The all-zero one-row case has zero BF16 error by construction. These are
operator measurements, not a full-model speed or quality claim.

Preserve `attention_fp4_canary01`: its first two arithmetic cases match the
quantized reference, but row-17 isolation reports failure because the harness
cloned a captured output before its first replay. Correct the snapshot by
replaying/synchronizing before cloning, then rerun the targeted native check.
Do not reclassify or delete the original receipt. Frozen weight processing
also returns early for already packed layers, preserving loader idempotence.

The serving trial reuses the selected 182,681.956 tokens/s warm c128 reference
and 146.9997 ms warm c1 reference, with the frozen 320-row development workload
and archived five-repeat score arrays. These are training-seen systems data,
not a final-ID selection set. Report pooled/source-macro AUROC, calibration,
partial AUC, ties, threshold flips and score drift; preserve strict failed
master-parity receipts separately from any finite diagnostic benchmark.
Keep one GPU worker and shared caches on the existing NC2 pod. The former
fusion API/engine 92464/92580 is identity-checked and retired before native work.

## Warm serving results

All fourteen initial passes complete, including eight full 320-row passes.
The c64 pair reaches 198,815/195,924 input tokens/s. The initial c128 pair is
192,690/164,249; retain the slower repeat and the original cold/first-touch
measurements. Five later c128 confirmation passes on the same resident worker
are 193,224 / 195,134 / 193,175 / 193,312 / 191,405, with no discarded samples.
They use the exact frozen 1,310,581 input tokens per pass. Compare with the
archived selected reference rather than replaying an unchanged control.

| Warm measure | Selected reference | Attention projections FP4 | Change |
| --- | ---: | ---: | ---: |
| c128 median input tokens/s | 182,682 | 193,224 | +5.77% |
| c128 total tokens / total time, all five passes | 182,300 | 193,243 | +6.00% |
| c1 p50, median of two passes | 147.00 ms | 156.82 ms | +6.68%, slower |
| c1 p95, median of two passes | 270.26 ms | 287.55 ms | +6.40%, slower |
| Source-macro AUROC | 0.884608 | 0.878553 | −0.6055 percentage points |
| Pooled AUROC | 0.908150 | 0.889729 | −1.8421 percentage points |

The aggregate uses total tokens / summed measured durations, not an arithmetic
mean of rates. AUROC comparisons use per-example medians over five repeats in
both conditions, across 29 dual-label sources and one undefined group. Mean/
maximum score differences are 0.052521/0.301638; margin differences are
0.453516/1.75. Twenty of 320 repeat-median decisions flip at threshold 0.5.
One baseline example is already threshold-unstable across its repeats.

Source-macro pAUROC@20 declines 0.778416→0.748985 and Brier worsens
0.139499→0.147939. Pooled pAUROC@20 declines 0.644971→0.599051 and Brier
worsens 0.130132→0.144583. Pooled recall falls 0.784810→0.753165 and FPR
rises 0.154321→0.197531; balanced accuracy falls 0.815245→0.777817.
Unique pooled scores change 87→84. Larger source-level losses include
insider-trading −13.33 AUROC points and soft-trigger −9.14 points. Some
four/six-row Qwen groups move by 16.67–25 points in either direction; retain
sample counts and avoid treating those small groups as stable estimates.
The quality change is material even though source-macro AUROC loses less than
one percentage point. These development measurements do not establish final-ID
performance or a generalization benefit.

The twenty-row canary passes the unchanged baseline-relative threshold:
mean score error 0.011250, correlation 0.999066. Strict master parity remains
failed, with master error/correlation 0.021275/0.996428; the nonzero recorded
adapter effect is 0.813632. Preserve that failed receipt separately from
finite/baseline-relative acceptance. Passing a small canary did not predict
the full-cohort ranking/calibration loss.

Loaded scope and runtime audits pass on engine **93625**: 64 MLP, 48 GDN and
16 attention FP4 projections, with BF16 outputs. Both new shapes dispatch
symbolic-M FROST plans; all sixteen actual attention layers are observed.
Combined preparation observes 128 calls (64 vendor, 32 norm, 32 SiLU), and
the native MXFP8 attention and original tuned projection audits remain current.
The runner also copies an old `native_swiglu.json` from retired engine 92580;
that unused inherited receipt is not a fresh fusion runtime pass. The broad
precision change is documented by the loaded and attention-projection audits,
not the legacy preparation field about unchanged weight/attention scope.

Keep the named attention-FP4 server warm as API **93539**, engine **93625**,
port **8010**, sole GPU ownership, approximately 168,886 MiB and 33°C at
collection. At collection the selected reference/default was unchanged. Preserve the
existing pod and shared caches. Artifacts include both native receipts and
exact sources, initial serving predictions, five warm confirmation repeats,
two warm c1 passes, runtime usage/source receipts, logs and an all-repeat
analysis under `results/b200_attention_gdn_serving/`. All **157** collection
hashes verify locally. Thirty-eight focused CPU tests and Ruff pass; the
scoped implementation is committed and the campaign records live PIDs.

## Subsequent reference selection

The user subsequently says, “Let's make it the reference now.” Select
`attention_fp4.json` for future B200 inference optimization, with the measured
quality and interactive-latency tradeoffs above explicitly recorded as
`user_accepted_finite`. Preserve the unchanged strict failed checks and all
historical controls. The existing worker is retained; selection changes only
the default entrypoint and checksum-bound client reference metadata. See the
[reference decision](../decisions/b200_attention_fp4_inference_baseline.md).
