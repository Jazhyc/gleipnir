# Qwen Triton mutation-analysis repair

The targeted repair removes the warning without changing the kernel source or
runtime launch. Keep it enabled in the warm named score-worker variant; the
selected generation reference remains unchanged. The completed screen observes
higher batch throughput, but does not isolate a full-model copy-removal mechanism.

## Cause and intervention

The active stack is Torch **2.11.0+cu130** with the isolated **Triton 3.7.1**
overlay, rather than the runtime's stock Triton 3.6.0. Torch's `generate_ttir`
marks every non-tensor argument as a constant during mutation analysis. Newer
Triton interprets these as genuine constexpr arguments and removes runtime
scalars from the IR signature while their argument attributes remain. The six
row-stride arguments of vLLM's fused Q/K RMSNorm/RoPE/gate kernel expose that
mismatch, producing `Function argument index out of range`. This matches the
[upstream report](https://github.com/pytorch/pytorch/issues/170049).

The [experiment contract](../../experiments/b200_mutation_analysis/README.md)
requires reproducing the error, precise mutation results, exact native parity,
unchanged inputs and changed-input graph replay before integration. The repair
is scoped to the source-bound Qwen kernel on this exact runtime. Construct the
constant map from declared constexpr arguments and `None`; retain runtime scalar
slots in IR argument order, then filter scalar results after tracing mutations.
Do not suppress warning logs, hard-code the writable buffers or upgrade the
runtime. Unknown kernel paths retain the original analysis and fallback.

The pinned kernel module SHA256 is
`3703af53dfef0c7e53035da2bee03ecada7fbd7698b40b05d6fd1ef9cf5077a6`;
Torch's analysis-module SHA256 is
`0486ea520f0c5b012d8d27adc3e6957a7cf47ac217fc7c5fd24f8a7c30175e6c`.
The adapter checks both, the runtime versions and its native validation receipt.
Original installed files remain untouched; installation patches the in-process
analysis hooks before compilation.

## Native evidence and failed probes

`mutation03` reproduces the original error and repairs IR generation. Analysis
returns exactly **q_out_ptr, k_out_ptr, gate_out_ptr**, rather than all nine
tensor arguments. Eight cases cover rows **1/17/129/1536/4096/32768** and
row-strided layouts at 17/4096. All outputs agree bitwise across eager and
original/repaired compiled execution; inputs remain unchanged, outputs finite,
and changed-input CUDA-graph replay is exact and actually changes the output.
Head geometry stays fixed, matching the model, while tensor sizes/strides are
dynamic. The native suite completes in **8.15 s** after initial setup.

Graph timings vary by less than 0.3% across versions. At M32768, original/fixed
medians are **583.98/583.61 microseconds**. Separate M4096 profiles show one GPU
kernel in each version and no extra clone/copy operators. Their timings are
excluded from speed claims. Thus the warning does not demonstrate extra GPU
copies in this isolated graph; full-model constraints may differ and remain
unattributed. The profiler emits its single-cycle event-retention warning;
each diagnostic deliberately captures only one cycle.

Preserve `mutation01` (stride-constexpr prototype) and `mutation02` (initial
bridge probe). Their native harness makes head/block geometry symbolic and
fails in the **unmodified control** with `SymNodeVariable` lacking `value`.
This does not establish that the stride prototype independently fails.
Use literal model geometry in the corrected harness, leaving tensor dimensions
dynamic. The final bridge repair avoids changing kernel annotations entirely.
No failed native candidate enters serving. Recovery reconstructs the verified
staged runtime from the recorded, identity-verified retired parent; it creates
no second resident server or new capacity.

## Integrated measurements, 2026-10-07

Use the same EU-RO-1 B200, GPU UUID and driver **580.178.04**. Retain the cached
two-logit endpoint, merged adapter, native Gigatoken/direct FROST, all FP4 scopes,
MXFP8 attention, BF16 state, causal LAST pooling, chunked prefill and prefix
caching off. The warning-free worker becomes ready in **250.61 s** under new
shared compile key `f0290e9cc3`. Its twenty-row accepted-reference score canary
passes with the same 0.00080652 mean error and 0.99996689 correlation as the
parent. Preserve inherited strict-master failures separately.

Reuse all archived controls; no baseline timing replay. Each concurrency has
one excluded full-cohort warmup and three c1/quick64 or six c128/full320 timed
repeats. Input/prompt identities, labels and exact token counts are bound.

| Warm median across all repeats | Previous score worker | Repaired worker |
|---|---:|---:|
| c1 p50 | 30.33 ms | 31.23 ms |
| c1 p95 | 133.68 ms | 128.54 ms |
| c1 input tokens/s | 96899 | 97272 |
| c128 input tokens/s | 207990 | 215707 |
| c128 p50 | 2329.48 ms | 2260.06 ms |
| c128 p95 | 2750.92 ms | 2615.34 ms |

Batch prompt throughput improves **3.710%** versus the previous score worker
and **2.211%** versus the selected cached generation control (211041 tokens/s).
Interactive median worsens **2.945%** versus the score parent; no interactive
latency improvement is demonstrated. Parent/new batch repeat ranges are
193671–210235 and 212562–217083 tokens/s. This is a sequential comparison, with
reconstructed launch environment after failure and no paired full-model
operator ablation: do not attribute the entire gain to this repair.

c1 scores match the score parent exactly. Repeat-median c128 mean/max score
differences are **0.002708/0.063962**. Source-macro/pooled AUROC changes
**+0.01619/+0.01172 percentage points** versus the score parent, and
**-0.42128/+0.21488 points** versus selected generation. One threshold flip is
on an example already unstable across parent repeats. Native bitwise agreement
and c1 parity support unchanged kernel arithmetic; altered batch schedules can
change FP4 results, but that cause is not isolated here. Per-source AUROC,
undefined single-label sources, calibration, ties and repeat diagnostics are
preserved. Full320 is training-seen systems development, not held-out quality
evidence or automatic reference/precision promotion.

## Closure

The final server log has **zero mutation-analysis warnings** through startup
and all benchmark requests. API/engine **18031/18054** remain healthy and warm
on localhost 8010; `cache_policy_state.json` records the repaired score variant.
The selected generation baseline is unchanged. All failed/native/integrated
receipts, source snapshots, traces, predictions, comparison metrics, retirement
and closure are collected under `results/b200_mutation_analysis/`. Shared
workspace compiler caches are retained. Focused tests give **15 passes** and
Ruff passes; no capacity lifecycle action occurs.
