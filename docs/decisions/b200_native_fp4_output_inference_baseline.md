# Direct FP4-output B200 inference reference

Selected by the user on 2026-10-07. Use
`experiments/b200_attention_gdn_serving/fp4_swiglu_native_output.json` as the
default B200 optimization recipe. This supersedes the
[attention-projection reference](b200_attention_fp4_inference_baseline.md).

Retain FP4 MLP, large GDN and full-attention projections, combined preparation,
selected shape-dependent output GEMM tiles and cuDNN MXFP8 causal prefill.
The MLP producer keeps the original path below 1,536 rows, uses symbolic-row
GEMM/SwiGLU with whole-row packing at 1,536 through 4,096 rows, and directly
emits packed FP4 with local block scales above 4,096. Thus it includes the
measured overhead improvements. BF16 outputs, recurrence, convolution, cache
and decode and FP32 gates/state retain their recorded scopes.

Five fully warm c128 passes give 196,866 input tokens/s (+1.88% against the
preceding reference, +0.67% against the overhead stage). Warm c1 median/p95
latencies are 159.46/275.92 ms; median latency is 1.68% slower than the preceding
reference. Source-macro/pooled development AUROC is 0.878105/0.885842, declining
0.0448/0.3887 percentage points. Ten of 320 repeat-median decisions flip at 0.5;
retain per-source, calibration and threshold results. These are archived-control
comparisons on training-seen systems development prompts, without final-ID
selection or a production traffic claim.

`experiments/b200_inference_benchmark/baseline.json` binds the original executed
recipe, completed sweep, frozen manifest, warm throughput and latency, native
validation and explicit `user_accepted_finite` receipt by SHA256. Archive the
preceding selection under `baselines/`. The twenty-row master-score canary
passes for this recipe. Strict changed-format baseline precision and inherited
MXFP8/vendor preparation precision failures remain false separately; user
acceptance does not convert them into numerical equivalence.

Reuse resident API/engine 95619/95657 on the existing NC2 pod, port 8010.
Selection changes client references, not loaded kernels or server settings.
Do not replay completed controls or restart the worker for selection/profile.
Stop the old server before subsequent changes requiring another GPU worker.
See the [full finding](../findings/b200_swiglu_overhead_native_output.md) for
negative receipts, row dispatch, source bindings and all measured tradeoffs.
