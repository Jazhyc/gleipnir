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

## Completed cuDNN FP4 result

All 64 MLP projections pass loaded precision audit; six native-GEMM decoded
reference checks have relative L2 0.00151–0.00196, below 0.01. No emulation or
non-MLP quantization is used. Loaded weight memory is 5.05 GiB, versus 5.96
for FP8 and 7.99 for BF16. Total serving allocation remains 49,914 MiB under
the fixed memory fraction. Fresh score canary passes: mean error/correlation
0.016473/0.998563 versus master, 0.015696/0.998663 versus dynamic LoRA and
0.018070/0.998321 versus merged BF16. This does not establish full-workload parity.

The initial six passes complete in `cudnn01`. Their medians are
21,102 / 51,532 / 95,282 input tokens/s at concurrency 1/4/16, below BF16.
First/repeat pass durations differ substantially: 16.950/10.240,
5.878/4.707 and 2.933/2.729 seconds. Keep these measured startup/first-use
effects; do not silently drop the first result or claim it fully warmed all
native plans despite the recorded generic startup warmup/autotuning.

A targeted replay on the same resident worker, `cudnn_warmed02`, reuses the
passed canary and already exercised full workload, without model reload or
another control run. Its exact driver is saved as `warmed_replay_executed.py`.
Two new repeats at each concurrency still show no speed gain:

| Concurrency | BF16 input tokens/s | FP8 input tokens/s | Warmed FP4 input tokens/s | FP4 throughput vs BF16 | Pooled AUROC BF16 → FP4 | Macro AUROC BF16 → FP4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 29,763 | 32,105 | 26,978 | −9.36% | 0.932512 → 0.920197 | 0.883207 → 0.858586 |
| 4 | 69,049 | 78,356 | 59,443 | −13.91% | 0.932512 → 0.921182 | 0.883207 → 0.877525 |
| 16 | 107,073 | 115,423 | 102,760 | −4.03% | 0.932020 → 0.926108 | 0.883207 → 0.876263 |

Each figure uses repeat medians, with ranking computed from per-row
repeat-median scores. All 64 rows and 269,411 tokens match the archived
baseline; AUROC macro covers 11 sources with both labels and retains 12
undefined single-label sources. Per-source results, repeat metrics, partial
AUROC, Brier, ties and thresholds remain in the JSON reports. At concurrency
1/4/16, pooled AUROC changes −1.23/−1.13/−0.59 percentage points and macro
changes −2.46/−0.57/−0.69 points. The same quantized serving layout varies with
batching; these small development groups do not establish population effects.

Warmed FP4 p50 latencies are 0.1325/0.2762/0.6418 seconds, versus BF16
0.1041/0.2240/0.5630. Mean/max paired score differences are
0.04197/0.32940, 0.04262/0.32940 and 0.03702/0.24075, with 4/3/2 threshold
flips. The initial suite's across-repeat/concurrency score ranges average
0.05623, maximum 0.36036, with three unstable decisions. Passing the twenty-row
canary misses larger full-workload drift; report it rather than promoting FP4.
This native cuDNN configuration is a negative result, not a claim about every
FP4 backend. Conversion/dispatch/plan costs have not been isolated by profiling.

Readiness takes 602.315 seconds, canary/workload startup is separate, and
four-length warmup takes 0.626 seconds. Local NumPy reads stall on the shared
filesystem; equivalent pod tests pass (17 focused checks, 143 seconds) and Ruff
passes locally. Preserve the first test-filename collision before its corrected
retry. Initial/warmed output coverage, finite scores, audit and every collected
artifact checksum verify. Collect logs/receipts/sources locally; no packed
weights are saved persistently. Only FP4 API PID 73839 / engine PID 73902
remains healthy and warm on localhost 8010; FP8 is stopped and pod retained.
