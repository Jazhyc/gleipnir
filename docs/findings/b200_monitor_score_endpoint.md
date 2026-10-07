# Dedicated two-logit monitoring endpoint

The endpoint works, but does not demonstrate a speed improvement over the
selected cached generation path. Keep it as a named scoring option; do not
replace the optimization reference or infer held-out quality parity.

## Contract and implementation

The [experiment](../../experiments/b200_monitor_score/README.md) freezes the
hypothesis, cached parent, score canary, training-seen quick64/full320 selection,
bounded three c1/six c128 repeats and stop conditions before launch. Reuse the
archived same-host `cache_policy_euro02/fresh_cached` controls; do not replay
the baseline. Each concurrency has one excluded whole-cohort warmup and fresh
HTTP clients per pass. Inputs, labels, model, adapter, precision and native
backbone kernels retain their recorded identities. Client queue implementation
differs slightly (semaphore tasks versus queue workers); this is an endpoint and
runner comparison, not an isolated projection-kernel experiment.

`POST /v1/monitor/score` uses vLLM 0.24.0's supported classification conversion,
causal LAST pooling and a two-row BF16 projection, returning raw logits,
`sigmoid(logit_1-logit_0)`, the margin and exact prompt-token count. It bypasses
generation, full-vocabulary projection, sampling, logprobs, ranks and text
decoding. Preserve KV/GDN state, chunked prefill, native Gigatoken/direct FROST,
the 32768-token step/context budgets and 128-sequence cap; prefix caching stays
off. The input is the complete rendered monitoring prompt, with no added
special tokens or truncation. The score server is a pooling worker; generation
clients cannot reuse it as though it were the prior completion server.

Worker audit verifies a **2 × 2560 BF16** bias-free head and exact byte agreement
with decision rows 15/16 in the merged tied embedding, SHA256
`17fee305103c071973e09276328df2bbdfecef76a9fca65c19d404f54fb7932c`.
The full vocabulary head is absent. Backbone validation receipts are unchanged.

## Measurements, 2026-10-07

`score02` runs on the same EU-RO-1 B200/driver as the controls. Initialization
takes **319.01 s**, including a new classification compile key `9e57e28563`
and 244.18 s compiling the graph. Shared workspace caches are retained.
The twenty-row canary passes against the accepted cached artifact: mean score
error **0.00080652**, correlation **0.99996689**, adapter effect **0.83990608**.
The parent's strict master failure stays separately recorded; this is not a
new strict-master pass.

| Warm metric, median across all repeats | Cached generation | Dedicated score |
|---|---:|---:|
| c1 p50 | 30.36 ms | 30.33 ms |
| c1 p95 | 116.11 ms | 133.68 ms |
| c1 input tokens/s | 102801 | 96899 |
| c128 input tokens/s | 211041 | 207990 |
| c128 p50 | 2405.42 ms | 2329.48 ms |
| c128 p95 | 2726.78 ms | 2750.92 ms |

c1 median changes **-0.088%**; its prompt-token throughput is **-5.742%** and
p95 worsens. c128 prompt-token throughput is **-1.445%**, with overlapping
repeat ranges (cached 167054–212697, score 193671–210235 tokens/s). Sequential
archived controls and host/run variation limit causal attribution. Removing
the identified 0.365-ms GPU output work does not ensure a wall-time gain:
the pooling frontend and runner have their own dispatch, copy and response
processing costs. Those costs have not been separately isolated.

c1 scores match within **8.89e-9**, with unchanged AUROC. Repeat-median c128
score mean/max differences are **0.006880/0.086512**, with no threshold flips.
Source-macro AUROC changes **-0.43747 percentage points**; pooled changes
**+0.20316 points**. The unchanged c1 scores support matching head arithmetic;
different batching can change FP4 arithmetic, but this run does not isolate the
cause of batch score differences. Per-source AUROC, calibration, ties, repeat
variation and undefined single-label sources are preserved in the comparisons.
The largest per-source delta is -12.5 points on the four-row varied-deception
Qwen-c subset (two examples per label), illustrating the small-source/tie
sensitivity of the macro metric. Nemotron's single-label source is undefined.
This development result is not a new quality acceptance or baseline promotion.

## Warnings and closure

Compilation repeats the cached baseline's Triton mutation-analysis warning:
PyTorch cannot generate/analyze the Q/K RMSNorm/RoPE kernel's intermediate
representation because of an argument-index error. Its documented fallback
marks all input tensors as potentially mutated, continuing conservatively;
it is not a numerical-corruption report. This may limit optimization or
introduce copies; no performance impact is attributed without an ablation.
See the [PyTorch 2.11 implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_higher_order_ops/triton_kernel_wrap.py).

The first `score01` launcher fails on a missing parent results directory before
retiring the reference server. Fix directory creation and preserve its failure
receipt. `score02` retires API/engine 16386/16409 before replacement. API/engine
**17261/17297** remain healthy and warm on localhost 8010, explicitly recorded
as `standard_cached_monitor_score` in `cache_policy_state.json`; the generation
baseline selection is unchanged. No capacity is created or terminated.

Artifacts are collected under `results/b200_monitor_score/score01/` and
`score02/`, including executed sources, input/reference bindings, canary,
all predictions, ranking/timing comparisons, precision audits, logs, retirement
and closure. Focused contract/route tests give **12 passes** and Ruff passes.
