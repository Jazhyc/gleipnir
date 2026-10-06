# Training-forward FP4 kernels in vLLM

Hypothesis: the validated training FROST forward path performs better in vLLM
than the first stock FlashInfer/cuDNN FP4 serving path. The user explicitly
requests this integration; the previous test did not use these kernels.

Preserve training arithmetic: hardware per-row activation packing, 16x16 weight
scales, fused descale with raw BF16 rounding, and symbolic-M native plans.
Reuse `pack_operand` and `Nvfp4ScaledGemm` unchanged through a forward-only
vLLM loading/custom-op adapter. Pack each merged MLP weight once, omit backward
copies and release original BF16 MLP weights. Use training-sized 16,384-row tile
planning with runtime M, caching two geometries instead of one plan per length.
Retain FROST frontend 1.31.0/cuDNN 9.26/CuTe DSL 4.8.0 and original source
receipts. The newer compiler/runtime is part of the intervention. Force
FlashInfer full attention and retain FlashInfer GDN and all non-MLP precision;
exposing the compiler overlay does not select FA4 serving attention.

Validate six layer-0 gate/up and down cases at M=1/17/129 against decoded
quantized FP32 arithmetic (<=1% relative L2). Require bitwise agreement with
the exact training `_native_linear` forward on the same merged weights/inputs;
release its temporary backward cache afterward. Audit all 64 MLP methods,
non-MLP BF16 linears, runtime/source versions and plan counts. Training gradient
waivers do not authorize inference score parity. This integration gets the
unchanged twenty-row serving canary. Finite failed score parity may finish the
bounded suite as diagnostic-only, preserving failure. Stop on arithmetic/
runtime/scope failure, nonfinite/missing score, truncation, input drift, OOM or
unsupported CUDA graph/plan launches.

Freeze the same 64 development prompts/labels, 269,411 tokens, order, binary
one-token scoring, two passes at concurrency 1/4/16, 32,768 context/token budget,
16 engine sequences, 0.25 memory fraction and prefix caching off. Include all
activation conversion and native launch costs. Compare speed, paired score/
margin drift, threshold flips and pooled/per-source/macro AUROC with archived
merged BF16, FP8 and stock FP4. Labels report drift and do not tune precision or
calibration. Macro covers dual-label sources, with undefined groups and sample
counts explicit. This training-seen workload cannot establish held-out quality.

Stop stock-FP4 API PID 73839 / engine PID 73902 before replacing it. Preserve
artifacts and the ephemeral merged checkpoint; use the existing NC2 B200 with
no capacity mutation. Reuse persistent compiler/kernel caches. Record readiness
separately from warmed throughput. If timed passes still show new-plan/kernel
startup costs, measure one bounded warmed replay on the same worker without
reloading a model or repeating the control. A >10% warmed gain merits follow-up;
no broader sweep or full ID rerun is requested. Keep the candidate resident.

```bash
python -m experiments.b200_frost_inference.run --output frost01
```

Results: `results/b200_frost_inference/`; logs:
`logs/runpod/b200_frost_inference/`. Check startup every 30–60 seconds in the
active session; no in-chat scheduler can promise wakeups after the turn ends.

Startup attempt `frost01` failed the runtime guard before loading weights:
the FA4 compiler overlay supplied Triton 3.6.0 ahead of the training pin. The
launcher now prepends the retained `/tmp/gleipnir-triton-3.7.1` target, matching
the training environment. Preserve the failed receipt; `frost02` uses this fix.

## Completed serving screen

`frost02` completes all six passes. All 64 MLPs use packed FP4 training-forward
methods, with BF16 non-MLP linears and two symbolic-M plans. The six native
checks match the original `_native_linear` bit for bit; decoded-reference
relative L2 ranges from zero to 0.00002036. The fresh twenty-row score canary
passes unchanged limits: mean difference/correlation versus merged BF16 are
0.019684/0.996641, versus master 0.018147/0.996695. Loaded weight memory is
5.2 GiB; overall serving allocation is about 49,300 MiB under the fixed fraction.

| Concurrency | BF16 input tokens/s | FROST input tokens/s | Throughput vs BF16 | FROST p50 latency | Pooled AUROC BF16 → FROST | Macro AUROC BF16 → FROST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29,763 | 27,509 | −7.57% | 0.1272 s | 0.932512 → 0.936453 | 0.883207 → 0.887626 |
| 4 | 69,049 | 64,601 | −6.44% | 0.2514 s | 0.932512 → 0.936453 | 0.883207 → 0.887626 |
| 16 | 107,073 | 124,595 | +16.36% | 0.4935 s | 0.932020 → 0.936453 | 0.883207 → 0.887626 |

Figures are medians of two post-startup passes; ranking uses per-row repeat
medians. Pass durations are 10.079/9.524, 4.196/4.145 and 2.184/2.141 seconds.
Readiness takes 622.566 seconds, with compilation and capture included;
HTTP warmup takes 0.400 seconds. No additional warmed replay is used: the
initial passes have no large startup cost like the earlier stock-FP4 trial.
P95 latency is 0.3077/0.3680/0.7588 seconds. Interactive p50 rises from BF16
0.1041 seconds, while concurrency-16 p50 falls from 0.5630 seconds.

At concurrency 16, throughput exceeds archived FP8 by 7.95% and warmed stock
FP4 by 21.25%. At concurrency 1/4 it remains 14.31%/17.56% below FP8 and only
1.97%/8.68% above stock FP4. The predeclared >10% performance-interest threshold
is met for batched throughput, supporting follow-up rather than adoption as a
universal serving default. Attention backends remain FlashInfer; no control
was rerun.

Pooled AUROC changes +0.394/+0.394/+0.443 percentage points; macro changes
+0.442 points across the 11 dual-label sources. Insider-trading AUROC falls
0.0625 and soft-trigger rises 0.111111; the other nine eligible sources stay
unchanged. Twelve single-label groups remain undefined. Mean/max paired score
error is about 0.0362/0.1927, with two threshold flips at every concurrency.
Across the six passes, mean/max score range is 0.001156/0.058688 with no
unstable threshold decisions, substantially below the stock-FP4 range.
Per-source/repeat metrics, partial AUROC, Brier, ties and thresholds remain in
the saved comparisons. The small training-seen set cannot establish quality.

This reuses the exact training **GEMM forward kernels on merged weights**.
Training adds separate BF16 LoRA contributions to the quantized frozen base;
here the merged base-plus-adapter weight is quantized once. Kernel bitwise
agreement does not imply identical whole-model training and serving arithmetic.
The cuDNN/compiler overlay is also part of the intervention. Compilation under
Triton 3.7.1 emits recoverable mutation-analysis warnings in vLLM's fused QK
normalization kernel;
PyTorch conservatively treats its inputs as mutated, and compilation/capture
finish. Their timing effect is not isolated. Do not attribute the full speed or
score difference solely to FP4 GEMM arithmetic without a matched ablation.

Thirteen focused CPU tests pass on the pod and Ruff passes locally. The local
shared filesystem stalls on cold imports, so retain the pod validation receipt.
Collect source archives, predictions, metrics, kernel/runtime audit, failed
startup and retirement receipts, verifying checksums and token identities.
Keep API PID 74857 / engine PID 75080 healthy and resident on localhost 8010;
the stock FP4 server is stopped and the existing B200 pod is retained.

The user subsequently selects this recipe as the B200 inference optimization
baseline, prioritizing high-concurrency throughput. The current config now uses
`baseline: selected`, resolved to the checksum-bound `frost02` results in
`../b200_inference_benchmark/baseline.json`. The executed historical config in
`frost02` remains immutable and retains its BF16 comparison. Existing server
arguments and arithmetic are unchanged; baseline selection needs no restart.
