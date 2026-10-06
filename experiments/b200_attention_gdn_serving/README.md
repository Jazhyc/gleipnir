# B200 attention and GDN throughput campaign

Current optimization reference: the user-selected **combined FP4 preparation,
shape-dependent output GEMM tiles and cuDNN MXFP8 prefill** stack. The default
entrypoint uses `fp4_gemm_tuned.json`. New comparisons use `baseline: selected`
and `high_reference: selected`; historical measurements remain intact. Use
182,682 input tokens/s for warm c128 comparisons and the separate 147.00 ms
warm c1 confirmation. See the
[current reference decision](../../docs/decisions/b200_tuned_fp4_inference_baseline.md).

## Gate/up GEMM and SwiGLU fusion trial

Hypothesis (2026-10-06): adapt NVIDIA's pinned SM100 block-scaled dense
SwiGLU kernel to remove its training-only gate/up write, preserve raw BF16
rounding and per-row FP32 descaling followed by BF16 rounding before SiLU,
and emit half-width BF16 activations for existing hardware NVFP4 packing.
Interleave frozen packed weight rows in 32-column up/gate blocks without
requantization. Compare the complete producer plus packing with selected
FROST gate/up plus fused SiLU/packing, including descale and graph replay.
Use native rows 1/17/129/1536/2304/4096/29184/32768, unchanged 1% relative-L2
admission, zero/extreme rows, isolation and changed-input replay. Bound tile
search before considering full serving integration. Stop on nonfinite output,
unsupported launches, or no native gain; preserve failures and restore the
selected worker. A viable candidate requires warmed serving throughput,
interactive latency and frozen 320-row development AUROC before selection;
reuse archived controls and never select on final ID. Keep one GPU worker,
the existing pod and shared caches. Whole-row amax remains in the separate
packing stage; this trial does not claim direct FP4 GEMM output.

Completed result: the corrected N192 kernel matches native down-projection
outputs and gives approximately 5% large-row producer gains. Warm serving
median improves 1.8%, but one slow confirmation repeat makes aggregate
throughput 2.1% worse, and warm c1 latency rises 6%. AUROC is effectively
unchanged. Keep `fp4_swiglu_fused.json` as a named experiment; the default
remains `fp4_gemm_tuned.json`. See the
[fusion finding](../../docs/findings/b200_fp4_swiglu_fusion.md) for all failures,
timings, precision receipts and the retained experimental worker.

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

## Native NVFP4 KV result

`nvfp4_attention02` completes fourteen timed passes. Eight native calls confirm
FP8 E4M3 queries, separate packed uint8 K/V views, FP8 E4M3 block-scale views,
causal D256 GQA and native TRTLLM prefill. The score canary fails: versus FROST,
mean absolute error 0.033490 and correlation 0.986485. All outputs remain finite.

| Concurrency | Input tokens/s | Gain vs FROST | Pooled AUROC delta | Source-macro AUROC delta |
| --- | ---: | ---: | ---: | ---: |
| 16 | 123,640 | −5.29% | +0.005235 | −0.014020 |
| 32 | 145,396 | −2.49% | +0.004454 | −0.014924 |
| 64 | 139,468 | −5.58% | +0.004044 | −0.014221 |
| 128 | 141,732 | −0.65% | +0.006603 | −0.013665 |

The c128 pair is 8.996/9.513 seconds. Full-cohort mean score drift is about
0.065, with maximum 0.532215. This candidate provides no observed throughput
benefit and reduces source-macro ranking despite a pooled increase; keep it
as diagnostic, without replacing the FROST/BF16 baseline. API 78611 and engine
78701 are retired before alternate GDN kernel checks.

`cutedsl_canary01` fails before native arithmetic: vLLM's `cute_utils` expects
`nvvm.Tcgen05GroupKind`, renamed `CTAGroupKind` by the pinned CuTe DSL 4.8.0
required by FROST. Preserve the receipt and checksum-verified executed sources.
The next attempt installs a process-local alias with verified CTA values
0/1; generated NVVM MMA/alloc/load signatures retain compatibility. Do not
change the validated FROST compiler install or upstream kernel source. Require
all six native output/state checks before a full serving launch.

`cutedsl_canary02` reaches first-kernel compilation, then fails on another DSL
rename: `tcgen05_commit_arrive` became `tcgen05_commit`. An AST audit of every
NVVM reference in vLLM's complete `cute_utils` package finds exactly these two
missing names; the commit call retains its address, multicast-mask, group,
location and insertion-point arguments. Extend the process-local compatibility
mapping and retry as `cutedsl_canary03`, retaining both failed receipts/sources.

`cutedsl_canary03` validates both process-local mappings in real native kernels:
the T=1 and T=17 output/state checks, FP32 oracle and continued state pass.
The strided-QKV case then fails in the reference normalizer's `.view` contract,
before that case reaches CuTe. Make Q/K contiguous before normalization in both
reference and candidate; retain strided V to exercise native layout support.
Retry the unchanged native CuTe arithmetic as `cutedsl_canary04` without
changing numerical ceilings or restarting any serving control.

`cutedsl_canary04` passes all six native cases, including strided V, ragged
isolation, continued state and the independent FP32 recurrence oracle. Output
relative-L2 is about 0.44–0.48%; final-state error about 0.25–0.35%, below the
unchanged 3% limits. Ragged split/packed results match exactly. Host-inclusive
calls at T4096 are 0.518 ms versus FlashInfer 0.466 ms; at [8192,8192], 0.586
versus 0.526 ms. These small native trials are slightly slower and do not
represent sustained high-concurrency serving. Run `cutedsl_gdn.json` on the
same whole-GPU envelope, preserving BF16 attention/GDN projections and FROST
MLPs. Bind the passed receipt and sources; require all 24 operators to select
CuTe without fallback, then report matched full-serving throughput and AUROC.

`cutedsl_gdn01` passes model loading, all 24 backend selections, the twenty-row
score canary and six quick timed passes. The full high-concurrency sweep then
returns NaN logprobs/HTTP 400 during its first c16 pass; no full-cohort timing
or AUROC is valid. Reject this backend at the production envelope despite the
passing native and small-model checks. Preserve the failed receipt, completed
quick predictions and server log; retire API 79972 / engine 80038 before a new
kernel. A monitoring gap leaves this failed trial idle until 10:48 UTC; this
is a monitoring failure, not additional benchmark work. Continue with the
separately specified FP4 GDN projection variant.

`fp4_gdn_projection01` tests the alternative FROST/cudNN forward GEMM family
on all 48 large GDN QKV/Z and output projections, after the modest W8A8 result.
Preserve the 64 FROST MLPs, small BF16 gate projections, BF16 convolution and
recurrence operands, FP32 gates/state and BF16 full attention. Require six
independent FP4 decoded-reference checks at M1/17/129 on the representative
projection shapes, complete dtype/method scope and the unchanged FROST MLP
checks. Use the same score canary, finite-output stop condition and fourteen
matched timing/AUROC passes. Quantization and conversion costs are included;
no baseline promotion follows a failed quality canary or nonfinite output.

`fp4_gdn_projection01` exits before model loading because the new adapter imports
`PackedNvfp4` from the packing module instead of its actual GEMM module.
Correct the import to `gleipnir.cudnn_fp4_gemm`, preserve the exited API receipt
and failure, then retry as `fp4_gdn_projection02`. The server wrapper records
successful FP8/FP4 registry imports before API/model startup in the same process;
this avoids another separate cold startup solely for the import check.

The rejected CuTe trial's completed quick 64-row medians are 28,555/62,242/
118,552 input tokens/s at c1/4/16: +5.70%/−4.46%/−1.00% versus the matched
control. At c16, quick pooled/source-macro AUROC deltas are +0.003448/+0.106692
on only eleven dual-label sources. These partial small-cohort metrics do not
supersede the NaN failure on the full workload; no full-cohort result or
production promotion is valid.

`fp4_gdn_projection02` completes fourteen timed passes. All 48 GDN projections
pass native decoded-reference checks (maximum observed relative-L2 0.00000753),
with representative weight errors 11.49% for QKV/Z and 17.98% for output.
All unchanged FROST MLP checks pass. The score canary fails: mean error/correlation
versus FROST 0.024425/0.992690; finite outputs remain mandatory.

| Concurrency | Input tokens/s | Gain vs FROST | Pooled AUROC delta | Source-macro AUROC delta |
| --- | ---: | ---: | ---: | ---: |
| 16 | 133,779 | +2.48% | +0.013694 | −0.007035 |
| 32 | 167,276 | +12.18% | +0.013322 | −0.006028 |
| 64 | 170,163 | +15.20% | +0.013772 | −0.006641 |
| 128 | 159,619 | +11.89% | +0.014690 | −0.004659 |

Full-workload mean/max score drift is about 0.053/0.50983, with 15–16 threshold
flips. Keep this faster variant diagnostic because its quality canary fails
and macro AUROC declines. Confirm the >5% c128 gain with five full-cohort passes
on the same warm worker, reusing the unchanged canary and native validation;
no new control server or model reload.

## Selected baseline and subsequent trials

The user explicitly accepts the FP4 GDN variant's 0.47 percentage-point c128
macro-AUROC decline and selects it as the inference stack baseline. Preserve
strict failed parity separately from checksum-bound `user_accepted_finite`.
Five complete warm c128 passes confirm median 165,927 input tokens/s (+16.31%
versus the previous control). A client transport error retains three valid
passes, then resumes the other two without restarting the GPU worker. All
nineteen complete arrays and seventy collected artifact hashes verify.

The default entrypoint now uses `fp4_gdn_projection.json`; `baseline.json`
binds the executed recipe, input manifest, full sweep, confirmation and quality
acceptance. Preserve the original FROST-only selection and all historical
references. Compare later kernels to this baseline with the same numerical
change limits, retaining strict master parity as a separate diagnostic and
mandatory nonzero adapter effect/finite output checks.

`fp4_gdn_fa4_01` changes only full attention to the installed causal paged BF16
FA4 backend, retaining all 112 FP4 projection GEMMs and BF16 FlashInfer GDN
recurrence. Require FA4 version 4, actual paged BF16 Q/K/V, causal D256 GQA
and the unchanged projection/native scope audits. Measure the same fourteen
passes and full-cohort ranking diagnostics against `fp4_gdn_projection02`.
FP8 attention and the validated forward-only FlashQLA recurrence are separate
subsequent candidates on this same selected FP4 stack; never revert GDN
projections to BF16 when claiming incremental gains over it.

`fp4_gdn_fa4_01` exits during startup because the native audit detects version 2,
not FA4. vLLM explicitly falls back for Blackwell head dimensions above 128.
Preserve that failed receipt; forcing version 4 alone cannot bypass the limit.
The already-pinned external FA4 4.0.0b33 interface has a dedicated SM100 D256
paged kernel. `fp4_gdn_external_fa4.json` adapts vLLM's forward call to that
verified package, keeps FP4 MLP/GDN fixed, uses 128-token page geometry and
single-split capture, and preserves causal masks, actual sequence lengths,
output buffers and LSE semantics. Reject unsupported features rather than
discarding them. Record external source/version and actual native dispatch;
require the same scope, score and complete throughput/AUROC checks before use.

`fp4_gdn_external_fa4_01` passes pinned version/source and all FP4 projection
checks, then fails at the first native forward because vLLM supplies descale
tensors for BF16 inputs. Its bundled BF16 FA4 branch ignores those tensors;
the external API rejects them. Preserve the terminal failure and exited-process
receipt. Remove these fields only in the BF16 adapter, require all layer scalar
descales to equal one, and retry as `fp4_gdn_external_fa4_02` with unchanged
math, workload, baseline, scope and numerical limits.

`fp4_gdn_external_fa4_02` clears the descale mismatch but exits before timing:
the hybrid allocator enlarges its 128-token logical pages to physical 640-token
pages, whereas native D256 FA4 requires 128-token TMA pages. Preserve its
terminal receipt and exited API/worker verification. The next intervention
exposes NHD K/V storage as smaller zero-copy views, including unused interleaved
K/V spans, and remaps the GPU page table. It leaves sequence lengths and causal
masks unchanged. First require shuffled-page payload equality, independent FP32
causal attention within 1% relative-L2, finite outputs and graph replay after
changing the input table. Then retry the unchanged fourteen-pass model screen;
measure the added page-table work in end-to-end throughput.

`fa4_paged_canary01` reaches the native forward but catches a return-contract
mistake: the pinned private API returns five values. Preserve that failure;
translate its output/LSE pair explicitly. `fa4_paged_canary02` passes all six
native checks, including shuffled physical pages and graph replay after table
updates, with maximum observed relative-L2 0.002253 against independent FP32
causal attention. Both K/V views share their original storage. The next model
trial, `fp4_gdn_external_fa4_03`, requires this source-bound receipt and leaves
the selected FP4 stack and fourteen-pass scoring envelope unchanged.

`fp4_gdn_external_fa4_03` reuses the passed six-case receipt and passes all
112 projection scope/native checks. Compilation completes (98.35 seconds),
then the engine's memory-profile forward supplies a page geometry rejected by
the adapter. Preserve that terminal failure; no score canary, timed passes or
model AUROC were completed. The isolated zero-copy D256 kernel is validated,
but full vLLM integration remains unsupported. Both API 82988 and worker 83104
exit and are verified absent before archiving. Keep the selected FP4/FlashInfer
baseline, source weights, master adapters and shared caches. The user has
returned; close this timed campaign with the accepted projection improvement.
Further FA4 integration needs actual profiling/cache shape and stride evidence
before another expensive startup. FlashQLA and combined FP4-projection/FP8-
attention configurations remain unrun candidates, so do not claim exhaustion
or terminate the retained pod on that basis.

## Forward-only cuDNN MXFP8 serving

The user requests a cuDNN MXFP8 attention trial on the selected FP4 MLP/GDN
baseline. Hypothesis: the native causal D256 kernel improves warmed prefill
throughput when backward preparation is omitted. Keep the existing BF16 paged
KV cache and all 112 FP4 projections. The pinned cuDNN paged MXFP8 path rejects
THD queries, so fuse BF16 cache gather with forward-only row-scaled K and
column-scaled V quantization, and use the native packed THD forward kernel.
Query quantization is row-scaled; softmax stays FP32 and output BF16. Include
gather, quantization, scale layout, allocations and native execution in timing.
Ordinary decode remains the original BF16 backend; the monitor emits one token.

Before model startup require exact payload/live scale atoms against the
previous validated NVIDIA producer, independent FP32 bottom-right causal
attention within the previous 5% native forward ceiling, finite outputs,
shuffled/interleaved cache pages, asymmetric query/history lengths, 128
sequences, 32768 context and graph replay with changed lengths/page tables.
Stop on structural/native/nonfinite failure. Then run the existing twenty-row
score canary and fourteen timing/AUROC passes with unchanged score limits.
Compare against `fp4_gdn_projection02` and its five-pass warm confirmation;
retain finite score failures as diagnostic, without accepting new quality
drift automatically. Confirm a >5% high-concurrency gain on the same worker.
Reuse archived controls and shared caches; one serving process owns the GPU.

`cudnn_mxfp8_canary01` passes producer byte agreement and the first full-query
native case (4.23% FP32 error), then fails the asymmetric query/history case
at 5.26%, above the unchanged 5% precision gate. All values are finite. Preserve
that failed receipt; do not widen its ceiling or promote this configuration.
The follow-up independently reconstructs row/column MXFP8 operands and online
128-key-tile FP8-P attention in FP32. Require <=1% agreement with this quantized
arithmetic reference, exact producer bytes, finite outputs and all envelope/
replay cases. Keep the original FP32 precision result separately as `passed`.
Only a source-bound `arithmetic_passed` result may support the already-enabled
finite diagnostic timing/AUROC screen; failed strict precision remains failed.

`cudnn_mxfp8_canary02` detects a reference-model mismatch at the same asymmetric
case. The pinned D256 kernel uses unit P scales (`SF_CONST_VALUE=0x7f`) and a
four-base-2-unit rescale threshold. The initial reference assumed the generic
cuDNN blog's 256 P scale and updated the maximum every tile. Correct only this
independent reference, retaining both historical failures and the 5% FP32 gate;
retry all declared cases as `cudnn_mxfp8_canary03`. Native arithmetic admission
is separate from precision acceptance. Full model timing remains diagnostic
if its native precision gate fails, with unchanged score/AUROC reporting.

`cudnn_mxfp8_canary03` still differs by 2.21% from the corrected quantized
reference on the long-history case. The pinned template mixes low-degree
polynomial exponentials into online softmax. Before a model timing run,
`cudnn_mxfp8_canary04` tests an isolated variant replacing those two mixed
helpers with hardware vector exp2 (`fastmath=True`). Keep unit P scaling,
the four-log2-unit rescale policy, all MXFP8 operands and both existing numerical
ceilings. Generate a checksum-named source under the shared cache; never mutate
the installed NVIDIA kernels or discard historical receipts. This variant's
performance includes any additional SFU cost; it is not a stock-kernel claim.

`cudnn_mxfp8_canary04` completes all seven cases, with exact producer payload/
live scale atoms and maximum quantized-reference relative-L2 0.001667. Both
changed-length/page graph checks pass. The unchanged FP32 precision gate remains
failed (maximum 0.053363). Keep `passed=false`, `arithmetic_passed=true` and
`diagnostic_only=true` separately. Twelve focused source/launch tests and Ruff
pass. `fp4_gdn_cudnn_mxfp8_01` starts the finite diagnostic model screen,
passes all 112 FP4 projection checks, compilation and initial profiling/warmup;
reuse this native receipt while checking actual serving dispatch and scores.

The first serving screen and five resident c128 repeats expose a metadata
integration error: pinned vLLM supplies cumulative **cache-page counts** in
`cum_seq_lens_kv`, whereas cuDNN's packed input requires cumulative token counts.
The initial synthetic fixture supplied token counts directly and missed this
contract. Preserve `fp4_gdn_cudnn_mxfp8_01` and `mxfp8_confirmation01` as invalid
integration results, not evidence about MXFP8 quality or performance. Retire
that server before changing kernels. The corrected bridge derives token offsets
on-device from the exact `seq_lens`, including partial final pages. Canary05
passes realistic page-count metadata, distinct token lengths, and changes both
on graph replay; workers reject earlier receipts without this contract. The
producer also takes batch count at runtime to avoid compiling every batch count.
Rerun all seven native cases and the model screen before interpreting scores.

The corrected `cudnn_mxfp8_canary05` passes all seven producer/arithmetic and
replay cases under the actual TRTLLM contract. Keep the strict FP32 precision
failure separate (maximum 5.336%); quantized-reference error stays below 0.167%.
`fp4_gdn_cudnn_mxfp8_02` completes all fourteen serving passes and passes the
baseline-relative score canary (mean error 0.012958, correlation 0.998448).
Strict master-score parity remains failed. Five additional resident c128 passes
in `mxfp8_confirmation02` give median 173,938 input tokens/s, +4.83% against the
selected baseline's five-pass 165,927. A transport failure after two complete
passes is preserved; the three missing passes resume on the same worker.
Source-macro AUROC changes +1.68 percentage points, pooled AUROC −0.37 points,
with twenty threshold flips. The baseline selection remains unchanged and the
experimental worker stays warm. See [the full finding](../../docs/findings/b200_mxfp8_serving.md)
for timing variation, calibration, invalidated results and artifact provenance.

The user subsequently accepts MXFP8 as the new optimization reference. Bind the
unchanged result and five-pass confirmation, native validation and a separate
`user_accepted_finite` receipt in `baseline.json`; archive the preceding FP4/BF16
attention selection. Preserve the failed strict native/score checks. Default
future launches to MXFP8 and bind selected full-cohort comparisons by checksum.
This changes reference metadata, not the running server's kernels/settings.

`mxfp8_profile01` profiles one full c128 pass on the same worker, excluding its
7.602 seconds from benchmark timing. Its 46,704 CUDA kernels sum to 6.696 seconds
and occupy 91.92% of the first-to-last-kernel window. FP4 GEMMs plus runtime
activation preparation take 34.82% of kernel time; fused elementwise/norm/gates/
layouts 26.66%; GDN core 12.80%; MXFP8 attention including preparation 10.46%;
remaining BF16 GEMMs 7.53%; causal convolution 5.90%; other kernels 1.84%.
FP4 packing/scales alone cost 12.01%, versus MXFP8 preparation 1.20%. Prioritize
FP4 row-scale/packing and producer fusion, preserving rounding/scales, before
another attention-conversion change. CPU operator totals overlap/nest and are
not wall-time fractions. Raw trace, table, predictions and exclusive kernel
membership remain archived; this is one batch-throughput profile, not c1 latency.

The follow-up source/trace investigation reconstructs all preparation shapes
without changing the resident worker. K9,216 MLP-down inputs account for 51.24%
of preparation time; the path scans BF16 activations twice and hides packing
inside a custom-op boundary. Hardware packing and GEMM output descaling are
already enabled. The prior fused-row experiment's high register footprint rules
out assuming a single large fusion will be faster. Installed FlashInfer 0.6.12
has a CUDA per-token NVFP4 quantizer with the required block/scale format; test
that bounded candidate first, using a host constant to avoid the wrapper's GPU
`.item()` synchronization. Producer fusion is a subsequent intervention because
the vendor norm/activation fusions use scalar scale contracts. See the
[detailed diagnosis](../../docs/findings/b200_mxfp8_serving.md#fp4-preparation-diagnosis-and-vendor-kernel-candidates).
No new kernel timing or AUROC claim follows from this inspection.

## FP4 preparation trials

The user authorizes testing all three proposed improvements. Hypothesis:
vendor per-token CUDA packing and fused SwiGLU/normalization producers reduce
activation preparation cost on top of the selected FP4/MXFP8 reference.
Preserve the FP32 master, merged model, 112 FP4 projections, MXFP8 prefill,
BF16 recurrence/cache/decode, token/scoring contract and shared caches. Retire
the existing server before changing active kernels; use one GPU worker.

First compare the installed CUDA per-token NVFP4 producer with the existing
hardware row packer at widths 2560/4096/9216 and rows 1/17/129/32768. Include
allocation, token-scale conversion and scale-layout work in timings. Check
decoded values/scales, zero and extreme rows, padding, isolation, finite native
GEMM outputs and changed-input CUDA graph replay. Record bitwise comparisons
separately from <=1% decoded-reference arithmetic admission. Then test row-scaled
SwiGLU packing and Qwen3.5 post-attention RMSNorm/residual packing, preserving
the recorded BF16 rounding boundaries, epsilon and weight-offset convention.
Require targeted producer/native checks before model startup; hard-stop on
structural, missing or nonfinite outputs. Never silently fall back.

For admitted candidates run the unchanged score canary, quick latency/full
throughput sweep and source/pooled AUROC against the saved selected MXFP8
reference, plus five warm c128 confirmation passes on the same resident worker.
Numerical finite failures may remain diagnostic under the existing explicit
trial policy, but do not overwrite strict receipts or promote new drift.
Reuse native/kernel caches and validation for unchanged components. No new
control server, calibration on final ID or new billable capacity is required.
Finish after the three independent variants and a combined variant if their
native/model results support it, then collect results and retain a useful worker.
No in-chat scheduling tool is available; startup and progress monitoring are
performed during this active turn and cannot promise a follow-up after it ends.

Vendor canaries01–04 retain a missing-PATH build failure, a scale-padding dtype
failure and the zero-row nonfinite failure. Add explicit zero payload/scale
initialization; never route zero rows to another producer. Canary04 then stays
finite but exceeds the unchanged 1% baseline decoded-precision comparison at
1.81% on M129/K2560. Preserve that failure. Canary05 separately compares native
FROST GEMM output with FP32 contraction of the actual vendor decoded operands,
retaining the 1% arithmetic ceiling, isolation and updated-input graph checks.
Only full arithmetic/finite admission permits the existing diagnostic model
screen; `passed` still denotes strict baseline precision and remains distinct
from `arithmetic_passed`. New source archives preserve the executed producers
and checking code. The eventual model AUROC check measures whether this native
rounding change is usable; no precision ceiling or historical result is widened.

Canary05 passes all twelve vendor arithmetic/isolation/replay checks, with native
GEMM relative-L2 at most 1.08e-5. Strict baseline precision remains failed at
1.81%. Long-row packing is 2.00/2.19/2.24 times faster at K2560/4096/9216;
these CUDA-graph measurements exclude HTTP and host dispatch overhead.

Fusion canary01 preserves the failed eager-style SwiGLU rounding attempt.
Inspecting generated Inductor code shows FP32 SiLU and multiplication followed
by a single BF16 output boundary. Canary02 matches this compiled contract and
passes eight cases for each fusion, including actual-operand GEMM arithmetic,
row isolation and changed-input graph replay. Select eight warps for SwiGLU
and four for normalization. At M32768, complete producer-plus-packing time is
0.806 to 0.457 ms for SwiGLU and 0.499 to 0.110 ms for residual/normalization.
SwiGLU payloads/scales are bitwise equal in the native fixtures; normalization
has small rounding differences and exactly preserves the BF16 residual output.
Model speed and AUROC must still be measured before interpreting these gains.

FlashInfer's existing 28 MB ephemeral cache is copied into the network-volume
`.cache/flashinfer` directory. Serving now sets `FLASHINFER_WORKSPACE_BASE` to
the repository root, preserving compatible native builds across pod restarts.

The independent vendor and SwiGLU screens each complete fourteen serving passes
and five warm c128 confirmation passes. Median rates are 181,786 and 180,903
input tokens/s, +4.51% and +4.00% against the selected 173,938 reference.
Vendor baseline-relative score parity fails; SwiGLU passes that canary. Their
source-macro/pooled AUROC changes are respectively −1.00/+0.47 and −1.83/−0.39
percentage points on the frozen training-seen cohort. Keep native fixture
agreement separate from whole-model scores. Both workers are retired before
the next kernel change; the selected reference remains unchanged. See the
[preparation finding](../../docs/findings/b200_fp4_serving_preparation.md).

The combined configuration requires separately bound receipts for all three
producers. Runtime usage must match 32 normalization fusions, 32 SwiGLU fusions
and 48 remaining vendor-packed GDN projections. The benchmark rejects failed,
stale-PID or incomplete preparation receipts before timing. Record the effective
FlashInfer cache path alongside the existing shared cache paths.

Normalization's five warm passes give 178,260 input tokens/s (+2.48%), with
source-macro/pooled AUROC −1.08/+0.05 percentage points. The combined stack
finishes all fourteen sweep passes and five warm confirmations at 185,767
input tokens/s (+6.80%). Its c1 median is 149.47 ms versus 149.50 ms; AUROC
changes −1.61/+0.81 points. The baseline-relative score canary passes, while
strict native/master failures remain recorded. The combined profile lowers
summed CUDA time from 6.696 to 6.326 seconds and launches from 46,704 to
44,688; exclude profiling from speed claims. All 76 timed passes have matched
IDs/prompts/token counts, finite scores and verified executed source hashes.

Final collection: `results/b200_attention_gdn_serving/fp4_preparation_collection01`.
The selected default stays unchanged. Combined API/engine **88949/89008**, port
8010, remain healthy and warm; `campaign.json` records completion, artifact
paths and retained-worker identity. Stop that server before changing kernels.
No new capacity is launched and the retained B200 pod stays running.


### Bounded FP4 GEMM tile tuning, 2026-10-06

Hypothesis: NVIDIA's forced N256 tile leaves performance available for the
four fixed projection shapes. Keep combined FP4 preparation, MXFP8 attention,
all model weights, packed operands and BF16 output rounding fixed. Sweep six
SM100 tiles/clusters in one persistent native worker using shared caches.
Measure rows 1/17/129/1536/2304/4096/29184/32768; select separately at M<=4096
and M>4096 only when every shape's numerical/zero-row/isolation/changed-input
replay checks pass and geometric-mean native gain reaches 2%. Reject unsupported
configs without fallback and preserve failures. Stop after this bounded sweep
and one integrated candidate. The incremental end-to-end reference is combined
preparation at 185,766.6 input tokens/s; also report the selected 173,938.1
reference. Use the unchanged frozen 320 training-seen systems-dev rows, one
excluded warmup plus five c128 passes, interactive latency and source/pooled
AUROC. This is kernel selection on systems-dev, with no final-ID promotion.
Stop the serving worker before native tuning or a changed serving configuration.


Native sweep `fp4_gemm_tune01.json` completes in 58.4 seconds. All 160
executable cases have zero relative-L2 difference, exact zero rows, row isolation
and changed-input replay agreement. Four M256/N256 builds fail the upstream
TMEM budget (496 columns available versus 512 required); preserve these rejects.
The forced N256/two-CTA tile wins every large-row shape and both input
projections. For M<=4096, N128/two-CTA wins MLP down by 13.60% and GDN output
by 5.70% (geometric mean across M129/1536/2304/4096). The integrated candidate
is `fp4_gemm_tuned.json`, with a source-bound native receipt, precompiled
symbolic-M plans and live per-shape dispatch audit. Its full-cohort reference
is the existing combined-preparation run; saved selected-reference scores are
also retained for the final report.

The integrated sweep completes fourteen passes. Five further warm c128 passes
have median 182,682 input tokens/s, -1.66% versus combined preparation and
+5.03% versus the selected MXFP8 reference. A fully warm c1 confirmation is
147.00 ms median / 270.26 ms p95, versus saved combined 149.47 / 286.38 ms.
The initial post-startup c1 median is 163.41 ms; preserve both measurements
rather than attributing all warm differences to the tile choice. Source-macro /
pooled AUROC is 0.884608 / 0.908150, -0.0584 / +0.0137 percentage points versus
the saved combined full-sweep repeat-median scores. Baseline-relative canary
passes; inherited strict master/preparation failures remain recorded.

The selected default is unchanged. The experimental server stays healthy and
warm: API/engine **89879/89962**, port 8010, same retained NC2 pod. Campaign
state is completed, with no active driver. Collection manifest
`fp4_gemm_tuning_collection01` verifies all 130 artifacts; 21 timed passes and
one excluded warmup have matched IDs/prompts/tokens and finite scores.
See [the tuning finding](../../docs/findings/b200_fp4_gemm_tuning.md).


Subsequent user selection, 2026-10-06: adopt the completed tuned stack as the
optimization reference. `baseline.json` now binds this run and its warm
throughput/latency confirmations plus explicit finite-quality acceptance.
Archive the preceding MXFP8 selection under
`baselines/frost_fp4_mlp_gdn_cudnn_mxfp8_prefill.json`. Preserve the original
executed configuration and all strict failures. Only the launch default and
client reference selector change; the existing API/engine 89879/89962 continue
with the same loaded arithmetic and capacity. No server restart or control
replay is needed. GEMMs remain about 32% of GPU kernel time in the saved
combined-preparation profile (24% FP4 plus 8% BF16); this is a pre-tuning profile,
not a new measurement of the selected server.


### Matched FP4 backend comparison, 2026-10-06

Hypothesis: a different FP4 GEMM implementation can improve the selected tuned
FROST stack beyond its tile sweep. Compare FROST, FlashInfer/cuDNN, CUTLASS,
TRT-LLM and CuTe-DSL on the same four projection shapes and eight native rows
1/17/129/1536/2304/4096/29184/32768. Preserve packed payloads/scales, scalar
weight inverse, per-row activation inverse and the two BF16 rounding boundaries.
Prepare backend-specific weight permutations once, outside timing, without
requantization. Time the whole raw GEMM plus required row-descaling/layout work,
using persistent plans, shared compiler/JIT caches and cached autotune choices.
CuTe tuning is bounded to at most eight representative supported tactics per
shape/row, with exclusions recorded. Keep unsupported/compile/numerical failures
rather than falling back silently. Native admission includes finite/error,
zero-row, isolation and changed-input graph replay checks at the unchanged 1%
relative-L2 ceiling. Select per projection and M<=4096 / M>4096 only after every
row passes and geometric-mean native improvement exceeds 2%.

Stop the selected API/engine before GPU trials; use one resident native worker
for the comparison. If a native winner survives, run one integrated candidate
against the checksum-bound selected reference, using the frozen 320 systems-dev
rows, warmed c128 throughput (182,682 input tokens/s), c1 latency (147.00 ms)
and AUROC/score diagnostics. Reuse unchanged preparation and attention receipts.
Stop after the bounded comparison and one integrated candidate; restore a useful
serving worker even if no backend wins. Do not start fusion/precision experiments
as part of this backend comparison. No final-ID promotion or new capacity.


Backend comparison completes without a selected alternative. Initial sweep
`fp4_backend_compare01` gives 128 passes and 32 preserved CuTe API failures in
253.1 seconds. A documented process-local register-tensor alias repairs the
CuTe/CUTLASS-4.8 mismatch; targeted `fp4_backend_compare02` gives 64 passes in
49.8 seconds. All four alternatives have zero observed numerical error in their
32 native cases, but no shape/row-band winner. At M32768, gate/up milliseconds
are FROST 0.6124, cuDNN 1.0327, CUTLASS 1.0267, TRT-LLM 1.2917, CuTe 1.0659.
These include required row descaling; the reference already fuses that epilogue.
Shared autotune cache retains 152 configurations. Preserve both exact source
archives and the original incompatibility; aliases are not installed in serving.

The reference is restored unchanged as API/engine **91253/91317**, port 8010.
Four bounded warmup requests progress and current-PID native audits pass.
Reuse the prior quality receipt and speed/AUROC controls; no new quality pass
or alternative model kernel is claimed. Collection `fp4_backend_collection01`
binds all native results, sources, installed implementation copies, cache and
restore receipts. See [the backend finding](../../docs/findings/b200_fp4_backend_comparison.md).
