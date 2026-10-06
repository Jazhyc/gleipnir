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
