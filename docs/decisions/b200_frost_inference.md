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

## Completed result

`frost01` fails before loading due to Triton 3.6.0 import priority. Prepending
the retained training Triton 3.7.1 target fixes the mismatch without relaxing
runtime checks. `frost02` passes all 64 MLP/non-MLP audits, six exact training
forward comparisons and unchanged twenty-row score-canary limits. Decoded
reference relative L2 is at most 0.00002036. Mean score error/correlation versus
merged BF16 is 0.019684/0.996641 on the canary.

Two post-startup passes at concurrency 1/4/16 produce median input tokens/s
27,509/64,601/124,595: −7.57%/−6.44%/+16.36% versus archived merged BF16.
Concurrency 16 exceeds archived FP8 by 7.95% and stock warmed FP4 by 21.25%.
P50 is 0.1272/0.2514/0.4935 seconds; BF16 is 0.1041/0.2240/0.5630.
The batched gain meets the predeclared performance-interest threshold; mixed
latency results do not support making this a universal inference default.

Pooled AUROC is 0.936453 at all three concurrencies, +0.003941/+0.003941/
+0.004433 versus BF16. Dual-label source macro is 0.887626, +0.004419.
Insider-trading decreases 0.0625 and soft-trigger increases 0.111111; nine
other eligible sources stay unchanged, with twelve undefined groups explicit.
Mean/max paired score drift is about 0.0362/0.1927 and two threshold flips.
Repeat/concurrency mean/max ranges are 0.001156/0.058688, with no unstable
decisions. Retain per-source ranking/calibration diagnostics; no population
quality or regularization claim follows from this small training-seen set.

The original forward GEMM is reused on merged weights. Training adds LoRA
separately; serving quantizes the merged weight. Keep that layout difference
explicit. cuDNN/compiler changes are part of this implementation, with
recoverable Triton mutation-analysis warnings in vLLM's QK normalization
compilation. Compilation/capture finish, but their timing effect has not been
isolated. A future kernel-only attribution requires a matched runtime ablation.
Readiness takes 622.566 seconds; pass pairs are close enough that no extra
warmed replay is used. Reuse controls, retain failures and keep the candidate
resident. Thirteen focused tests pass on the pod and local Ruff passes.
