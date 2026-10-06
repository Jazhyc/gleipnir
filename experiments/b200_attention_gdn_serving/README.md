# B200 attention and GDN throughput campaign

Started 2026-10-06 03:19 UTC. The user authorizes approximately nine hours of
attention/GDN optimization, prioritizing throughput at high concurrency, FP8
first, alternative kernels and small integration changes. Stop the old server
before changing kernels; keep one serving engine on the GPU. Capacity is already
authorized. If sensible options are exhausted, the user authorizes terminating
the B200 after verified artifact collection. No public deployment is requested.

Hypothesis: native low-precision attention and projection kernels improve
complete monitor input-token throughput after FROST FP4 MLP optimization.
The selected control is `results/b200_frost_inference/frost02`; preserve its
timings/predictions and validated source/runtime/merge receipts. Use the same
FP32-master-derived ephemeral merged BF16 model, frozen prompts and one-token
binary scoring. Retain BF16 non-target operations, no prefix caching, and
network-volume compiler caches.

First establish the FROST control at a production-sized envelope: 90% GPU memory,
128 engine sequences, 32,768 scheduled tokens and 32,768 context. This envelope
differs from the recorded 25%/16-sequence control and needs one matched baseline;
reuse it for all compatible candidates. Include the original quick 64-row,
two-repeat c1/4/16 screen and a full 320-row two-repeat c16/32/64/128 load sweep.
The latter is the original systems cohort (1,310,581 prompt tokens), with frozen
seed-0 order and labels, sufficient requests to sustain the higher concurrency.
Client concurrency and engine sequence capacity are separate; report both.
Record the highest measured sustainable throughput, all repeat durations and
latency percentiles. A concurrency increase alone is not a kernel speed gain.

Candidate order, revised when evidence warrants:

1. FlashInfer/TRTLLM FP8 Q/K/V full attention and FP8 KV storage, with FROST MLPs
   and BF16 GDN fixed. Verify actual FP8 query/cache dispatch and causal D256 GQA.
   Freeze and record appropriate scales; count packing/conversion costs.
2. Alternative native attention kernels, including NVFP4 KV with FP8 query and
   compatible BF16 FA4, if the first attention kernel does not help. Preserve
   causal ragged semantics and paged cache/graph compatibility.
3. Native FP8 GDN QKV/output projection GEMMs, leaving small gate projections,
   convolution and recurrence operands BF16 and gates/state FP32. Test FP4
   projections separately if warranted by native kernel availability.
4. Alternative GDN prefill backends (FlashQLA and CuTe DSL) and targeted fusion,
   dtype/conversion or dispatch changes guided by profiling. An FP8 recurrence
   needs kernel support; do not advertise FP8 state storage as FP8 computation.
5. Combine only measured improvements, then examine remaining bottlenecks and
   capacity/token-budget tuning at the strongest high-concurrency condition.

Before a new precision/backend trial, inspect startup closely, audit loaded
scope/runtime and actual attention/GDN dispatch, retain finite/missing-output
checks and perform the unchanged adapter-specific twenty-row score canary.
Reuse unchanged FROST arithmetic validation when possible; check changes in
the supported batching envelope. Keep a finite failed score canary as diagnostic
and report the failure without widening limits. Stop on nonfinite/missing
outputs, native arithmetic/scope failure, source/input drift, truncation, OOM
or unsupported graph launches. Diagnose and preserve failed attempts before
retrying; an observation timeout alone never justifies restarting a live run.

Every condition reports pooled/per-source AUROC, equal-source macro over
dual-label sources, undefined groups, repeat variation, score/margin drift,
threshold flips, ties, calibration and partial AUROC. Bind labels to exact
prompt identities. These training-seen development sets do not establish
held-out quality; no final ID promotion or silent acceptance of degraded
serving precision. A >5% matched high-concurrency gain merits confirmation;
retain smaller and negative results. Warmed confirmation reuses the same
worker and passed unchanged canary rather than reloading or rerunning a control.

Write a condition configuration, intervention and matched reference before each
launch. Archive executed code/configs, receipts, predictions, metrics and logs;
commit completed features and findings after focused tests and successful
startup. Collect important artifacts after each phase and before termination.
Keep source weights/master adapters/caches on the network volume; merged models
and reconstructable packed weights can remain ephemeral.

Check startup every 30–60 seconds during active goal work, then progress and GPU
health at least every ten minutes. This session has no timed agent heartbeat
scheduler; active goal turns support checks but cannot promise timed wakes after
a turn ends. Record live PIDs, configuration and artifact paths for continuation.

Outputs: `results/b200_attention_gdn_serving/`; logs:
`logs/runpod/b200_attention_gdn_serving/`. All conditions use the existing NC2
B200, no concurrent control server or newly launched capacity.

## Whole-GPU FROST control

`high_base01` completes six quick passes and eight full 320-row passes. Its
canary matches the archived FROST scores exactly (mean error 0, correlation 1).
All 64 MLPs pass the FROST scope/native audit. Eight observed native attention
calls have BF16 Q/KV, causal masking, D256, 16 query heads and four KV heads.
The native paged view uses 16-token pages; do not infer its layout solely from
the hybrid cache's physical allocation. Startup takes 620.651 seconds, HTTP
warmup 0.438 seconds. Serving uses about 169,374 MiB of 183,359 MiB GPU memory.

| Client concurrency | Full-workload input tokens/s | p50 latency | Pooled AUROC | Source-macro AUROC |
| --- | ---: | ---: | ---: | ---: |
| 16 | 130,542 | 0.504 s | 0.889534 | 0.889694 |
| 32 | 149,113 | 0.830 s | 0.889690 | 0.889694 |
| 64 | 147,705 | 1.707 s | 0.889963 | 0.889848 |
| 128 | 142,661 | 3.189 s | 0.889397 | 0.889292 |

Two-repeat medians; the workload is 320 rows and 1,310,581 prompt tokens, with
29 dual-label sources and one undefined source. This is a different population
from the quick 64-row screen; do not compare their absolute AUROCs. The engine
allows 128 active sequences at all client loads. Peak observed throughput is
at c32; c128 is slower despite a larger queue. Preserve both peak and
highest-concurrency measurements. The c128 pass pair is 8.913/9.478 seconds,
so modest gains require additional warmed confirmation rather than assuming
perfect measurement stability.

Thirteen focused CPU checks and Ruff pass. All fourteen prediction arrays,
native scope/dtype receipts, prompt/token identity and finite logprobs verify;
43 collected artifact checksums match locally. The control API 76454 / engine
76507 is retired before the FP8 candidate, with its log and identity-bound
retirement receipt preserved. No control remains on the GPU.

`fp8_attention01` starts with native FP8 Q/K/V and the same 90%/128-sequence
envelope. Its requested `--calculate-kv-scales` flag is explicitly disabled by
vLLM for this hybrid model: startup calibration has uninitialized recurrent
state. Effective initial scales are unit values, not a completed real-prompt
calibration. Retain this requested/effective difference; inspect canary/AUROC
and scale state before deciding whether a real-prompt calibration is needed.

## Native FP8 attention result and profile

`fp8_attention01` completes all fourteen passes. The canary passes: mean error
versus FROST is 0.010543, correlation 0.999104; master error/correlation are
0.016524/0.997552. Eight native calls confirm FP8 E4M3 Q/KV, D256, causal GQA,
and the paged prefill kernel. Effective scales are all 1.0, with BMM1 scale
0.0625 and BMM2 1.0, consistent with the runtime's disabled calibration flag.

| Concurrency | FP8 input tokens/s | Gain vs matched FROST | Pooled AUROC delta | Source-macro AUROC delta | Threshold flips |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 135,093 | +3.49% | +0.000820 | −0.010495 | 14 |
| 32 | 154,689 | +3.74% | +0.000449 | −0.010476 | 14 |
| 64 | 152,512 | +3.25% | +0.000547 | −0.010282 | 14 |
| 128 | 143,166 | +0.35% | +0.000840 | −0.009048 | 13 |

The c128 repeats are 8.775/9.568 seconds, so its small median gain is not
convincing. Full-workload mean/max paired score error is about 0.0314/0.2861,
despite the passing small canary. Preserve the source-macro decline and avoid
promoting this configuration solely on pooled AUROC or a modest c32 gain.

A separate resident c128 profile completes, excluded from benchmark timing.
Its 38,976 CUDA kernels total 7.940 seconds; their interval union is 7.934
seconds over an 8.509-second first-to-last-kernel span (93.24% active). Kernel
sum shares: BF16 GEMMs about 27%, FROST including packing about 20%, GDN core
about 12%, causal convolution about 5%, and native FP8 full attention about 9%.
CPU/GPU overlap means CPU operator totals are not wall-time fractions. Do not
double-count nested profiler operators or call all gaps CPU dispatch. Trace,
profiler tables and raw kernel-name/duration summaries remain in the artifacts.

This prioritizes FP8 GDN QKV/Z and output projection GEMMs next, with BF16
attention, convolution and recurrence retained and small gate projections
unchanged. Require six quantized-reference checks on the two representative
GDN projection geometries (M=1/17/129), native Cutlass W8A8 dispatch, complete
48-projection coverage and all unchanged FROST checks. Also test another native
attention path rather than treating the first FP8 kernel as an endpoint.

## Native FP8 GDN projections

`fp8_gdn_projection01` completes fourteen timed passes. All 48 large GDN
projections use native Cutlass W8A8; small gate projections, full attention,
convolution and recurrence retain BF16, with FP32 gates/state. Six independent
quantized-reference checks give zero observed BF16-rounded output error;
representative weight reconstruction error is 2.64%. All unchanged FROST
checks pass. The score canary fails its existing correlation limit: versus
FROST, mean absolute error 0.021591 and correlation 0.994822. This is diagnostic
only, despite improved AUROC on this small training-seen cohort.

| Concurrency | Input tokens/s | Gain | Pooled AUROC delta | Source-macro AUROC delta |
| --- | ---: | ---: | ---: | ---: |
| 16 | 135,404 | +3.72% | +0.008302 | +0.005239 |
| 32 | 153,391 | +2.87% | +0.008380 | +0.005919 |
| 64 | 154,542 | +4.63% | +0.007521 | +0.005181 |
| 128 | 145,290 | +1.84% | +0.007716 | +0.005428 |

The c128 repeats are 8.616/9.465 seconds; its small gain remains inconclusive.
Full-workload mean score drift is about 0.039 and maximum 0.29845. Preserve
per-source regressions even when macro/pooled averages improve. API 77510 and
engine 77576 are retired before changing kernels.

The next isolated candidate is native NVFP4 KV storage with FP8 queries and
block scales, unchanged BF16 GDN and FROST MLPs (`nvfp4_attention.json`). Audit
actual packed uint8 KV, FP8 queries, block-scale tensors and causal D256 GQA.
The native path returns FP8 attention output then converts to BF16; include
that conversion in complete timings. This is not a claim that every attention
operation uses FP4. Run the same score canary and fourteen-pass matched sweep;
retain finite failures as diagnostic and never promote nonfinite outputs.

Before a new GDN prefill backend, `gdn_canary.py` compares outputs and final
V-first FP32 state against FlashInfer at native D128, 16 Q/K heads and 32 value
heads. Freeze relative-L2 limits at 3% for both; preserve nonfinite failures.
Use lengths 1, 17, 129, ragged [1,127,513], 4096 and [8192,8192], nonzero initial
states, a sequential FP32 oracle, split/continued state and ragged isolation.
Record warmed host-inclusive kernel-call times; these are feasibility checks,
not complete serving throughput. Run them only after retiring the serving
process, then require the usual model score and runtime dispatch checks.

`nvfp4_attention01` reaches readiness, then fails its first HTTP canary because
our audit assumed a combined cache tensor. Native NVFP4 supplies separately
strided K/V pairs (and separate block-scale views). No valid throughput or
AUROC exists for that attempt. Preserve the failure and exited-process receipt;
correct tuple handling with focused tests, then retry as `nvfp4_attention02`.
This integration failure is not evidence that the native kernel is unsupported.
