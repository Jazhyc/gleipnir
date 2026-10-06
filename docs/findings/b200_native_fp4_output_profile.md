# Direct FP4-output serving bottlenecks

Measured 2026-10-07 on the retained NC2 B200, API/engine 95619/95657, port
8010. The user selects the [direct FP4-output reference](../decisions/b200_native_fp4_output_inference_baseline.md).
Reuse the warm server, compiler/native plans and archived controls. No kernel
change, restart, new GPU worker or final-ID evaluation occurs.

## Bounded measurement

One Torch-profiler pass of the frozen 320-row, 1,310,581-input-token workload
at client concurrency 128. Prefix caching remains disabled. The instrumented
HTTP pass takes 6.724902 seconds; exclude it from speed comparisons. The saved
five-pass warm reference remains 196,866 input tokens/s and separate c1
median/p95 remains 159.46/275.92 ms. This is a batch-prefill profile, not a
measurement of interactive latency bottlenecks.

44,784 CUDA kernels sum to 5.852121 seconds. The union of their intervals is
5.850683 seconds over a 6.425253-second first-to-last-kernel window: 91.06%
kernel coverage and 8.94% gaps. This is timing coverage under instrumentation,
not tensor-core utilization, uninstrumented dispatch cost or achieved hardware
peak. The interval outside that kernel window also includes client/server work.

| Exclusive GPU kernel category | Seconds | Share of summed kernel time |
| --- | ---: | ---: |
| Fused FP4 gate/up GEMM, SwiGLU and direct packing | 1.3389 | 22.88% |
| Normalization, gates and layout elementwise kernels | 1.2917 | 22.07% |
| Other FP4 GEMMs | 0.8926 | 15.25% |
| GDN core | 0.8015 | 13.70% |
| MXFP8 attention core | 0.5965 | 10.19% |
| Causal convolution | 0.3831 | 6.55% |
| Separate FP4 preparation, packing and scales | 0.3011 | 5.14% |
| MXFP8 gather, quantization and metadata | 0.0783 | 1.34% |
| Other kernels | 0.0768 | 1.31% |
| Remaining BF16 GEMMs | 0.0423 | 0.72% |
| KV cache stores and metadata | 0.0407 | 0.69% |
| Sampling and output scoring | 0.0066 | 0.11% |
| Medium-row fused gate/up GEMM and BF16 SwiGLU | 0.0020 | 0.03% |

Categories use kernel names; `breakdown.json` retains every exclusive member,
count and duration. Fused producer time includes arithmetic and packing and
cannot be attributed entirely to matrix multiplication. All GEMM-containing
categories together account for **38.89%**. Correlating the 4,064 standalone
FP4 GEMM launches with their innermost custom CPU operation matches every
launch: `frost_inference_linear` accounts for 0.599156 seconds (10.24% of all
kernel time), and `packed_frost_linear` for 0.293481 seconds (5.01%). These
identify general projection and packed MLP routes, without claiming a separate
per-layer projection breakdown.

Direct-output kernels launch 1,280 times, with 32 medium-row producer launches
and 32 small-row SiLU-pack launches. This profile therefore predominantly
measures the large-row direct route. Its largest individual kernel is the fused
producer. Largest elementwise kernels include QK norm/RoPE/gating (0.196091 s),
the compiled GDN normalization/gating expression (0.187229 s), and post-conv
processing (0.169071 s). Separate normalization-plus-FP4 packing costs
0.154006 s; vendor quantization/per-token scaling costs 0.129506 s. Packing
inside the fused producer is counted once, in the producer category.

## Implications

GEMM-containing work remains the largest family; the fused producer is the
largest single kernel. Converting the remaining BF16 GEMMs addresses only
0.72% of this measured kernel time. The promising next investigations are
the fused producer's tiling/epilogue and the sizeable normalization/gating/layout
work, followed by GDN. Attention core is 10.19%; separate FP4 preparation is
5.14%, so neither is the dominant remaining component. These shares are
prioritization evidence, not predicted end-to-end gains from another precision
change. Hardware stall, bandwidth and tensor-core counters have not been
measured; do not call the producer compute-bound or memory-bound from this
trace. Producer-plus-down native measurements already showed smaller complete
MLP gains than isolated producer gains.

CPU `aten::copy_` sums to 2.4173 seconds; `frost_inference_linear` sums to
1.4531 seconds. CPU operators nest and overlap GPU execution, and profiler
overhead is present. These sums are not additional wall-time fractions and
do not establish a CPU bottleneck. The 8.94% GPU-window gaps likewise include
all reasons for gaps, not only CPU dispatch.

## Identity and collection

Artifacts: `results/b200_attention_gdn_serving/native_fp4_output_profile01`,
including predictions, raw compressed trace, profiler table, exclusive category
membership, launch-scope mapping, executed analysis/client scripts, source
binding, selected baseline, health and campaign receipts and file checksums.
All 43 live bound kernel/source hashes match local files; executed recipe and
frozen manifest match the selected reference.

Trace SHA256: `9a93d6c441419532e0f6abe3124848fda72a85ab1557026572ac87e6035074cb`.
Breakdown SHA256: `a4bcc007cbe3ea0a58daec62728ec60a69ba55aef0c5495e541050060309a941`.
Reference-selection SHA256: `39811c43e0b4bbf574e682d7b21f09e394909af3af4a69f3b398193cace89166`.

The first profiling-client invocation fails before profiling starts because its
Python path omits the repository root. Preserve that log. The corrected client
completes the pass and stops profiling successfully. Post-profile health is
HTTP 200; sole GPU worker 95657 uses approximately 170,340 MiB at 32 C and
is idle. The server and pod remain available. Fifteen focused selection/overhead/
native-output tests and Ruff pass; selection commit is 165f864. No scheduling
tool is available; no after-turn monitoring is promised.

## Kernel-window gap follow-up

The remaining 574.570 ms (8.942% of the kernel window) is not all GPU idle.
Intersect the union of kernel-free intervals with GPU memcpy/memset intervals,
and correlate every following kernel with its CUDA runtime/driver launch.
Merged-interval recomputation independently verifies the copy overlap.

| Exclusive timing partition | Milliseconds | Share of GPU window |
| --- | ---: | ---: |
| GPU copies/memsets during kernel-free intervals | 88.135 | 1.372% |
| Before the next kernel launch starts, excluding GPU copies | 445.550 | 6.934% |
| Remaining launch/execution spacing | 40.886 | 0.636% |

All 39,917 gaps match their next kernel's launch. The second partition measures
when the host submits work; it does not isolate scheduling, Python/custom-op
preparation, synchronization or profiler overhead. The last partition includes
time in launch calls and spacing after submission. These instrumented totals
are not guaranteed recoverable speedup in an unprofiled worker.

36 gaps exceed 1 ms and 1,265 span 100 us to 1 ms; these account for 71.99% of
all gap time. Thus most gap duration is not the accumulation of tiny
inter-kernel spacing. Innermost engine-thread scope overlap highlights
`frost_inference_linear` (87.726 ms), GDN (34.922 ms), packed FP4 GEMMs
(34.114 ms), copying and execute-context work. Those CPU scope overlaps include
copy intervals, are temporal associations rather than causal attribution, and
are separate from the exclusive table above.

Device-to-device copies sum to 81.146 ms. Correlation places 81.009 ms and
1,008 copies inside `vllm::qwen_gdn_attention_core`: one output copy per GDN
layer per scheduled batch. They total 257,670,709,248 bytes, exactly
1,310,581 tokens times 4,096 output elements times two BF16 bytes times 24
GDN layers. There are 936 copies of 256 MiB for 32,768-token batches; the
remaining 72 copies match the three partial batches. The installed
`qwen_gdn_linear_attn.py` ordinary-prefill caller omits the destination
argument and assigns `core_attn_out_non_spec` into `core_attn_out` afterward.
That caller assignment causes the measured copy; the chunk wrapper also has
an optional copy when supplied a destination. This confirms output copying,
not full recurrent-cache copying. It consumes roughly 1.26% of the GPU window and is a concrete
future opportunity if the kernel can write directly into the destination.

Additional artifacts: `gap_analysis.json`, `gap_cpu_overlap.json`,
`gap_copy_scopes.json`, and the installed GDN source. No new profile, kernel
change or serving restart is performed for this follow-up.

## Engine gaps versus frontend/tokenizer work

Follow-up on the saved reference trace: the 7--8% figure measures kernel-free
time around host submission and launch, not CPU utilization or a separately
measured tokenizer cost. All recorded CPU events belong to engine PID 95657;
`ignore_frontend=true` excludes API PID 95619 and tokenizer tracing.

Intersecting kernel-free intervals with the union of engine execution-context
annotations puts 570.120 of 574.570 ms (99.23%) inside those annotations, with
4.450 ms outside. This is temporal overlap, not a causal attribution to any
specific operator. GPU annotations show 39 full 32,768-token batches, one
29,116-token batch and two small batches (1,428/2,085). Together these findings
argue against tokenizer starvation explaining most of the recorded gaps.
They do not rule out frontend cost before the kernel window, in HTTP latency
or at other workload sizes/concurrencies.

The profiled pass takes 6.724902 s, versus a warm unprofiled median of
6.657211 s (about 1.02% longer). This is a single profiled pass against archived
warm repeats, not a controlled profiler-on/off attribution. It does not support
calling the entire 7--8% instrumentation overhead. Engine dispatch, custom-op
preparation, allocation and synchronization remain candidates; instrumentation
can amplify them. CPU scope sums overlap GPU execution and are not additional
wall-time percentages. A matched text-versus-exact-token-ID request comparison
would isolate frontend tokenization/serialization contributions without changing
GPU kernels; it has not been run here.

Additional diagnostic: `native_fp4_output_profile01/gap_execution_context.json`.
No new serving run, kernel change or server restart is performed.

## Prefill graph coverage

The recorded recipe captures at most 256 tokens. Every profiled batch exceeds
that limit, and installed dispatch returns graph mode `NONE`; the trace shows
one individual launch per kernel and no graph-launch calls. See the
[prefill-dispatch assessment](b200_prefill_dispatch.md) for pinned-source
evidence, bounded graph-capture candidates and padding/memory limitations.
