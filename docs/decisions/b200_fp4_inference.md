# Native cuDNN FP4 MLP inference screen

Decision date: 2026-10-06. The user requests FP4 MLP inference after the native
FP8 screen, and explicitly asks to report AUROC deviation with every kernel
change. Reuse the existing NC2 B200 and completed merged BF16/FP8 predictions.
Stop the old FP8 API PID 73262 / engine PID 73329 before starting the replacement.
Keep checkpoint, FP32 masters, caches, receipts and previous results; no capacity
allocation/termination, control rerun, full ID run or scheduling sweep.

Use the pinned vLLM native FlashInfer/cuDNN NVFP4 GEMM backend, with a narrow
online MLP loading adapter because the pinned stock NVFP4 loader expects a
quantized checkpoint. Pack only 64 fused MLP projections once at loading, retain
BF16 elsewhere, and dynamically compute activation ranges during forward.
Use the stock scale swizzle/padding/packing/GEMM; bind source hashes into the
compiler key. Validate sample reconstruction and six independently CPU-decoded
quantized matmul cases before model-level diagnostics. Preserve unchanged
canary limits and explicitly retain finite parity failure when completing the
user-requested bounded AUROC/speed diagnostic. Fail closed on wrong scope,
native arithmetic failure, nonfinite output or structural/runtime failures.

The experiment README freezes the same 64 prompts, 269,411 input tokens, labels,
order, budgets, prefix-cache policy and two passes at concurrency 1/4/16.
AUROC reporting binds labels to exact prompt identities and includes pooled,
per-source and equal-source macro scores over groups with both labels, paired
deltas and repeat variation. Include undefined sources, sample counts, ties,
partial AUROC, Brier and threshold diagnostics. Backfill the previous FP8 trial
from its archived predictions without another GPU run. The small training-seen
set is a development diagnostic, not held-out production quality or a promotion
criterion. Record the ongoing AUROC preference in `AGENTS.md`.

## Completed negative cuDNN result

Native implementation checks pass (six relative-L2 errors 0.00151–0.00196)
and the fresh twenty-row score canary passes. Mean/correlation versus merged
BF16 are 0.018070/0.998321. All 64 fused MLPs use the required cuDNN kernel,
with BF16 elsewhere. Weight memory drops to 5.05 GiB; total server allocation
remains 49,914 MiB at the fixed memory fraction.

Initial passes have substantial first-use effects. Preserve `cudnn01`, then
run a targeted warmed replay on the same server with unchanged inputs/caches,
reused parity and no control rerun. Warmed throughput at concurrency 1/4/16 is
26,978/59,443/102,760 input tokens/s, still 9.36%/13.91%/4.03% below BF16 and
below FP8. Interactive p50 rises 0.1041 → 0.1325 seconds. No performance-interest
threshold is met. Do not diagnose conversion or CPU dispatch as the cause
without profiling, or generalize this one native backend to all FP4 kernels.

Warmed pooled AUROC changes 0.932512 → 0.920197, 0.932512 → 0.921182 and
0.932020 → 0.926108. Source-macro changes 0.883207 → 0.858586/0.877525/0.876263.
Macro covers 11 dual-label sources; 12 undefined groups remain explicit.
Paired score mean/max differences reach about 0.042/0.329, with 4/3/2 threshold
flips. The initial six-pass mean/max score range is 0.05623/0.36036 with three
unstable decisions. The small canary passes while full-workload score/ranking
drift remains; retain this negative result and do not adopt the configuration.

Seventeen focused CPU tests pass on the pod; Ruff passes locally. Local NumPy
filesystem-read stalls require this validation fallback; preserve the earlier
test-filename collision before correcting it. Collect all twelve initial/warmed
passes, AUROC/backfill reports, source/merge/cache receipts, kernel audit and
logs, verifying coverage and checksums. FP4 API PID 73839 / engine PID 73902
remains warm and healthy on localhost 8010, the previous FP8 server stopped.

The stock FP4 server is subsequently retired for the user-requested
[FROST training-forward comparison](b200_frost_inference.md), which records
the new resident worker. Preserve this stock result and its retirement receipt.
