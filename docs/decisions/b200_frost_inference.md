# Reuse training FROST forward arithmetic in vLLM

Decision date: 2026-10-06. The user requests testing the exact optimized FP4
training forward kernels after the negative stock FlashInfer/cuDNN serving
result. That earlier result does not measure this implementation.

Use unchanged `pack_operand` hardware per-row conversion, 16x16 weight scales,
`Nvfp4ScaledGemm` fused row descale and symbolic-M plans through a narrow
forward-only vLLM custom op. Pack only forward weights once at loading, releasing
BF16 MLP weights and omitting backward copies. Use 16,384-row plan geometry for
the training-sized tile choice while supporting actual M at launch. Reuse
the checksum-bound training source generation and retained frontend 1.31.0,
cuDNN 9.26 and CuTe DSL 4.8.0. Record the runtime difference as part of the
intervention; keep FlashInfer attention/GDN and all non-MLP arithmetic unchanged.

Before benchmarking, require six native decoded-reference cases and bitwise
agreement with the original training forward on the same merged weights and
inputs at M=1/17/129. Audit all 64 MLPs and release temporary backward cache from
these checks. Preserve the original score-canary limits and report finite
failures as diagnostic when completing the authorized small AUROC/speed suite.
Do not reuse a training loss/gradient waiver as serving quality acceptance.

The experiment README freezes the same 64 rows, token/sequence budgets,
one-token score, two repeats at concurrency 1/4/16 and disabled prefix caching.
Include conversions in timings and record pooled/per-source/macro AUROC deltas,
ties, calibration and threshold changes against archived BF16/FP8/stock-FP4.
No control rerun, held-out quality promotion, full ID run or broad tuning sweep.
Stop old API PID 73839 / engine PID 73902 before replacement. Keep source
adapters, ephemeral merged checkpoint, caches, logs and results. Use the existing
NC2 B200 with no capacity mutation. Active-turn monitoring checks startup every
30–60 seconds; no scheduling tool is available for later wakeups.
