# Direct GDN prefill output on the Direct FP4 reference

Date: 2026-10-07. The user requests removal of the GDN output copies found
in the [reference profile](b200_native_fp4_output_profile.md). Hypothesis:
passing the existing destination to FlashInfer removes temporary output
allocation and its copy into vLLM, improving serving speed without arithmetic
changes. Use the selected Direct FP4 reference and archived controls: 196,866
input tokens/s, c1 median/p95 159.46/275.92 ms, macro/pooled systems-dev
AUROC 0.878105/0.885842. No final-ID selection or new capacity.

## Intervention and admission

FlashInfer 0.6.12 already exposes an optional output tensor in its public
`chunk_gated_delta_rule` API. vLLM 0.24.0's ordinary-prefill caller omits the destination argument,
then copies the allocated result into `core_attn_out`. The adapter passes a contiguous, aligned
view of the caller's active destination through that public API. Preserve
vLLM's BF16 Q/K normalization and contiguous preparation, exp of FP32 log
gates, FP32 beta/state conversions, V-first state and final-state behavior.
Do not replace the native GDN algorithm. Reject invalid dtype/device/stride,
insufficient or unaligned destinations and operand aliasing; require that
FlashInfer returns the supplied destination. A missing destination retains
the allocating API behavior. Bind installed FlashInfer and vLLM source hashes.

Retire identity-verified API/engine 95619/95657 before native checks. Keep
the existing pod, merged model, master adapters and shared caches. Native
checker PID 96696 completes all seven cases: 1/17/129/641/4096/16384/32768
tokens, including ragged sequences and strided V. Output and FP32 final state
are bitwise equal to the original wrapper at every size. Destination alias,
untouched prefix/tail, immutable initial state, normalization-disabled and
final-state-disabled routes, optional destination, zero inputs, isolated
sequences and changed-input CUDA-graph replay pass. The GPU checks take
20.9368 seconds after package startup.

Native receipt: `results/b200_attention_gdn_serving/gdn_direct_canary01.json`,
SHA256 `9c56cca76668ff5305b4c0be492fecfc24f5ec8b092b394b941a59d5c3425654`.
Six exact source snapshots are collected and hash-verified. Twenty-five focused
CPU tests and Ruff pass. Commit 1e7d13b adds the recipe, adapter, audited worker,
source-bound native checker, safe buffer-contract tests and startup validation.

## Serving measurement

The corrected `gdn_direct_output.json` keeps the selected Direct FP4 stack.
API/engine 97318/97421 remain the sole serving/GPU worker on the existing
NC2 pod, port 8010. The failed first attempt is recorded below. Live direct
writes and caller adaptation cover all 24 GDN layers. The twenty-row HTTP
canary passes and exactly matches the selected reference (mean score error
zero, correlation one); master error/correlation remain 0.019734/0.996962.

All fourteen initial sweep passes complete. Preserve post-startup c128 rates
199,252/153,989 tokens/s separately from the fully warm confirmation. The
unchanged selected reference stays the default; no final-ID selection occurs.

| Warm measure | Selected reference | Direct GDN output | Change |
| --- | ---: | ---: | ---: |
| c128 median input tokens/s | 196,866 | 196,780 | -0.04% |
| c128 aggregate input tokens/s, five repeats | 196,857 | 196,199 | -0.33% |
| c1 median latency | 159.46 ms | 168.85 ms | +5.89% |
| c1 p95 latency | 275.92 ms | 291.07 ms | +5.49% |
| Source-macro development AUROC | 0.878105 | 0.878066 | -0.00391 pp |
| Pooled development AUROC | 0.885842 | 0.885822 | -0.00195 pp |

Five warm c128 rates: 197,013 / 195,968 / 198,559 / 192,770 / 196,780.
Keep every repeat and the excluded warmup; controls are archived, without an
interleaved replay. No clear throughput benefit is established, and latency
regresses. Repeat-median mean/max score differences are 0.001162/0.062419,
margin differences 0.016016/0.500000, with one 0.5 threshold flip on
`bash_arena:54269d226838b17554af4f9f`. Only bash-arena source AUROC changes
(-0.001134); the tiny overall ranking shift does not imply identical scores.
Preserve all repeat, calibration, pAUROC, tie and per-source diagnostics.
Source-macro pAUROC/Brier are 0.751796/0.151917; pooled values
0.596810/0.147839. Pooled recall/FPR are 0.753165/0.191358 and unique scores
fall 85 to 84. Native output/state equality and the identical twenty-row
canary support arithmetic preservation; different full-serving batch shapes
are observed, so this screen does not establish identical HTTP outputs.

## Caller wiring correction

Preserve failed `gdn_direct_output_serving01`: wrapper adaptation alone does
not activate direct writes because the actual ordinary-prefill caller omits
`core_attn_out`. Its live audit rejects missing direct dispatch from all 24
layers before timing. API/engine 96864/96982 are retired, and their log/source
receipts remain archived. This also corrects the earlier profile's attribution
of the output copy to the optional wrapper copy; the active copy is in the
parent caller's final assignment.

Generate an isolated Apache-2.0-licensed adaptation of the source-bound
installed `_forward_core` method. Change exactly one chunk-call keyword and
one final assignment: forward the active destination and omit the final copy
for ordinary prefill. Mixed prefill/decode, speculative and decode-only routes
retain their original output merging/copy behavior. Reject already-modified
or structurally different source and bind original/generated hashes; do not
edit installed vLLM. End-to-end CPU caller tests confirm direct destination
reuse, untouched tails and no final copy, plus unchanged fallback routes.
Thirty focused tests and Ruff pass. Reuse the unchanged direct-output native
admission; the kernel arithmetic and buffer-contract adapter are unchanged.
Retry artifacts use `gdn_direct_output02_serving01`.

## Copy removal and interpretation

A bounded profiled c128 pass confirms **1,008 GDN output copies to zero** and
**257,670,709,248 bytes to zero**. Only 84 unrelated device-to-device copies
remain, totaling 10,486,080 bytes. The public FlashInfer destination is used;
no ordinary-prefill final copy remains. Instrumented pass time (6.868840 s)
is excluded from speed claims.

Summed GPU kernel time is 5.872923 s versus 5.852121 s in the reference
profile. GDN kernel time is 0.824319 s versus 0.801464 s; most other GPU
families are nearly unchanged. Kernel coverage is 89.93% over a 6.529001 s
window, versus 91.06% over 6.425253 s. Kernel-free intervals increase to
0.657510 s from 0.574570 s despite removal of the output copies. Thus removal
of this copy does not produce an end-to-end win in the measured recipe.

These are single instrumented profiles, with 41 large direct-FP4 producer
batches plus one small route in the candidate versus 40 large, one medium
and one small in the reference. CPU scope sums grow for several unrelated
operators too; they overlap/nest and cannot isolate the added guards or establish
a causal host penalty. Hardware counters are absent. Do not attribute the
regression solely to alias checks, output-buffer caching or any unmeasured
hardware effect, and do not treat saved copy time as a promised speedup.

Keep this as a measured named candidate. Reference/default selection remains
unchanged because warm throughput does not improve and interactive latency
regresses. Retain the healthy candidate worker for compatible follow-up work.

Artifacts: `gdn_direct_output02_serving01`,
`gdn_direct_output02_confirmation01`,
`gdn_direct_output02_latency_confirmation01`,
`gdn_direct_output02_profile01`, and `gdn_direct_output_collection01` under
`results/b200_attention_gdn_serving/`. Native fixtures and exact installed
source snapshots remain in `gdn_direct_canary01`/`gdn_direct_canary01_sources`;
the failed first serving attempt and retired logs are preserved.

Trace SHA256: `15967433e34580d9cbb045b3366285e55decea8c5000d5a84322bf80ea9bde5e`.
Generated caller SHA256: `d9788af07554f30f7ad42224ce7675acf0c3dc18bf391788490f11c9ca893d3d`.
All 47 live bound source hashes, frozen prompt IDs/checksums, finite predictions
and collected campaign artifacts are verified. Copies and analysis scripts
are archived with checksums. Thirty focused tests and Ruff pass. Implementation
commits: 1e7d13b and 9e99e0e. Profiling is stopped; server health is HTTP 200,
sole GPU worker 97421 uses approximately 170,360 MiB at 32 C. No new capacity
or after-turn monitoring is scheduled.

## Deferred performance path, 2026-10-07

The user asks to leave this unresolved if the likely upside is small. Defer
further direct-output tuning; preserve the implementation, failed attempt,
receipts and warm candidate. No server or capacity lifecycle action is requested.
Keep the selected Direct FP4 reference unchanged.

The observed 81.009 ms copy budget is 1.26% of the reference's 6.425253 s
kernel window. Removing only that cost while holding everything else fixed
would improve kernel-window rate by 1.28%, or instrumented HTTP-pass rate by
1.22%. This suggests roughly 1--2% potential throughput gain, allowing for
small unmeasured allocation/dispatch savings; it is not a measured speedup
or a hard upper bound on all possible integration changes. Recovering the
candidate's 9.39 ms median-latency regression would first restore the baseline,
rather than establish a gain over it.

At observed large-copy bandwidth, byte-linear estimates are 0.031 ms of copy
work for the quick cohort's median 502-token prompt, 0.260 ms for its mean
4,210-token prompt and 1.776 ms at its 28,733-token maximum. These are GPU-copy
estimates, not request-latency predictions; small transfers, launch/allocator
cost and overlap can differ. There is no evidence here for a substantial
interactive-latency improvement over the selected reference. Resume only when
isolating wrapper/dispatch cost is useful to a broader optimization effort.
