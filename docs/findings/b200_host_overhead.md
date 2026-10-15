# Short-request host overhead

2026-10-15, EU-RO-1 B200, FP8 default, C1. Short requests are host-bound. A
1K-token step took about 32 ms of engine host time against about 8 ms of GPU
work. Two adopted, exact or near-exact changes cut short C1 latency by about
5 ms (~15%). The remaining ~27 ms of host time per step is diffuse.

## Diagnosis

The [probe](../../experiments/b200_host_overhead/README.md) launches the
default through normal startup, with ptrace-permitting diagnostic workers. It
times each `execute_model` and samples both processes with `py-spy`. Per-step
host time is about 27–32 ms regardless of size (1K–8K). There are no blocking
syncs; the cost is roughly 1,100 Python-dispatched launches per step. The
[kernel profile](b200_kernel_profile.md) shows 81% GPU idle at 1K.

## Changes and paired evidence (same server)

| Change | Status | Short C1 effect | Numerics |
| --- | --- | --- | --- |
| [Length-gated GDN route](b200_gdn_route.md): non-CP below 8K tokens per call | adopted | −2.3 ms (quick64 p50 34.1 → 31.8) | canary MAE 0.0005; ≥8K bitwise |
| [Lean GDN host path](../../experiments/b200_gdn_fast/README.md) | adopted | −2.9 ms (1K 32.4 → 29.5) | bitwise; full 1K–1M sweep `full02` equals prior runs |
| Strided gated norm in the lean GDN path (no gate copy) | adopted | 16K/32K C1 −6.8%/−4.2%; c128 throughput +4.5% | bitwise |
| [Fused GDN conv1d + post-conv](../../experiments/b200_gdn_fast/README.md) | not adopted | 1K/quick64 ~2 ms slower; 8K–32K within noise; c128 −1 to −3% | bitwise |
| [Lean custom-op dispatch](../../experiments/b200_lean_dispatch/README.md) | not adopted | within noise (≤0.7 ms) | bitwise |

Startup now verifies that the gate and the fast path are active. With both
adopted (`host03`/`host04`), the 1K client p50 is about 30 ms. Engine step is
26.6 ms, and the API (HTTP, tokenization, IPC) adds about 3.4 ms. Full320 c128
throughput is unchanged or slightly higher.

## Remaining host time (`host04`, 1K step ≈ 26.6 ms)

| Component | Host time per step |
| --- | ---: |
| GDN layers: conv1d, post-conv and norm launchers, fast-path glue, FlashInfer call | ~8.1 ms |
| Compiled-graph glue and op dispatch | ~7.6 ms |
| Engine loop outside `execute_model` (scheduler, IPC, waiting for the next request) | ~4.4 ms |
| FROST FP4 linear wrappers | ~4.1 ms |
| Attention op (MXFP8 producer and plan Python, 8 layers) | ~2.1 ms |

No single item exceeds about 2 ms of removable host time with exact,
host-only changes. A large further reduction would need CUDA-graph replay of
whole short prefill steps, including the eager GDN and attention ops. An
earlier [large-prefill graph trial](b200_prefill_graph_capture.md) captured
only the compiled pieces, gained 1.9% at c1 and changed scores through padding.

## Measurement caveat

At short lengths, separate server instances differ by about 2 ms, and the
startup c4 canary is not reproducible across instances because batch
composition changes scores by up to 0.053. Short-latency and exactness
comparisons need same-server paired toggles. All results above use them.
