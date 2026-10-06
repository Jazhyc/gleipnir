# Native FP4 MLP serving on B200

Hypothesis: native FlashInfer/cuDNN NVFP4 MLP GEMMs improve merged-model HTTP
throughput beyond BF16 and FP8, with an acceptable diagnostic AUROC difference.
Use the existing NC2 B200, stop the previous FP8 serving process before starting
its replacement and retain its completed measurements. Allocate no capacity.

Freeze the same 64 prompts, 269,411 input tokens, order, seed, one-token binary
HTTP score, two repeats at concurrency 1/4/16, prefix-cache-off policy, 32,768
context/token budgets, 16 engine sequences and 0.25 memory fraction. Primary
control is the archived merged BF16 run; also compare with archived FP8 MLP
results. No control rerun, full ID pass or scheduler tuning. This is a small,
training-seen systems workload; labels report ranking drift and never calibrate
quantization or select a prompt, precision allocation or production recipe.

Only the 64 fused projections in 32 MLPs use FP4 weights and activations. Full
attention/GDN projections/recurrence, embeddings and vocabulary head remain BF16.
Load the immutable merged BF16 checkpoint from ephemeral disk and pack MLP
weights once on GPU, with CUDA E2M1 codes, E4M3 scales per 16 values and FP32
global scale. Compute dynamic activation global range at every invocation and
include its reduction, native packing and GEMM in all timed results. No activation
calibration or new persistent weight checkpoint. Keep FP32 masters untouched.

The pinned vLLM lacks an online NVFP4 loader, so add a narrow loading adapter
that calls the stock native kernel selected by `--linear-backend flashinfer_cudnn`.
Preserve its scale swizzling/padding and GEMM implementation. Pass dynamic
activation inverse-scale/descale tensors through an immutable view; do not
mutate model parameters during forward. Reject emulation or a different backend.
Bind custom arithmetic/source hashes into the compiler cache key through
`additional_config`; retain shared compiler/kernel caches across restarts.

Before model timings, validate finite source weights/scales, sample weight
reconstruction relative L2 <=0.25 and six native-GEMM cases (layer-0 gate/up
and down at 1/17/65 rows) against independent CPU-decoded FP32 quantized matmul,
requiring <=0.01 relative L2. Audit all 64 packed MLP methods/kernels and BF16
elsewhere. These validate implementation, not agreement with original BF16.
Run the unchanged twenty-row score canary against archived master, dynamic LoRA
and merged BF16 (mean error <=0.02, correlation >=0.99, nonzero adapter effect).

The user asks for AUROC deviation with every kernel change. After a finite
but failed score canary, retain its failure and finish the bounded 64-row suite
as **diagnostic only**, to obtain AUROC and timing without promoting the candidate.
Do not widen the canary limits. Stop on native arithmetic/scope failure,
nonfinite/missing output, provenance drift, truncation, OOM or backend/compiler
failure. Report pooled and per-source AUROC plus macro over sources with both
labels, paired deltas against the same baseline, repeat metrics, score/margin
drift, threshold flips, ties, calibration and source sample counts. Undefined
single-label sources remain explicit. Passing a small canary or unchanged AUROC
does not establish population quality. A >10% warmed gain merits follow-up.

```bash
python -m experiments.b200_fp4_inference.run --output cudnn01
```

Results: `results/b200_fp4_inference/`; logs: `logs/runpod/b200_fp4_inference/`.
Keep a functioning candidate resident for compatible checks. Startup checks run
every 30–60 seconds during the active turn. No in-chat scheduler is available
to promise agent follow-ups after this session.
