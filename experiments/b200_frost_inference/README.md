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
