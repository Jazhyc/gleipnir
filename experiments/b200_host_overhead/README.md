# Host overhead of short single requests

Diagnostic, user-requested 2026-10-15: why short C1 requests take about 35 ms
when the [kernel profile](../../docs/findings/b200_kernel_profile.md) shows
about 8 ms of GPU work at 1K. No recipe change or promotion.

`probe.py` launches the adopted default through the normal canary-guarded
startup. It swaps in two diagnostic variants: a server and worker that call
`prctl(PR_SET_PTRACER, ANY)`, so `py-spy` can attach under Yama scope 1
without `CAP_SYS_PTRACE`, and a worker that records each `execute_model` wall
time. The probe times sequential C1 requests (1K and 8K synthetic, quick64).
It then samples the engine and API processes with `py-spy` (500 Hz,
non-blocking, raw collapsed stacks) during a stream of 1K requests.
`--gdn-cp off` sets `GLEIPNIR_FLASHINFER_GDN_CP=off` for the diagnostic server.
That is FlashInfer's non-context-parallel GDN prefill route, which the vLLM
0.31 migration screened.

```bash
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_host_overhead.probe --name host01 [--gdn-cp off]
```

## Results (EU-RO-1 B200, adopted default)

| Workload | Client p50, CP auto | Engine step, CP auto | Client p50, CP off | Engine step, CP off |
| --- | ---: | ---: | ---: | ---: |
| 1K synthetic | 36.1 ms | 32.2 ms | 31.2 ms | 28.0 ms |
| 8K synthetic | 51.2 ms | 29.6 ms | 50.7 ms | 26.5 ms |
| quick64 (real, 188+ tokens) | 35.6 ms | 31.1 ms | 32.2 ms | 28.3 ms |

Each forward step costs about 28–32 ms of host time regardless of size, against
about 8 ms of GPU work at 1K. There are no blocking syncs: one device-to-host
copy per step, and 43 `item()` calls on CPU tensors.

`py-spy` attributes CP-auto engine samples per 1K step as follows:

| Component | Share | Host time per step |
| --- | ---: | ---: |
| vLLM GDN core custom op (24 layers) | 39% | ~12.7 ms |
| Compiled-graph glue | 20% | ~6.3 ms |
| Outside the forward (engine loop, scheduler, IPC) | 16% | ~5.1 ms |
| FROST FP4 linear wrappers | 12% | ~3.9 ms |
| Attention op | 6% | ~1.9 ms |
| FP4 SwiGLU, FP8 linear | ~7% | ~2.2 ms |

Inside GDN, FlashInfer's context-parallel prefill does host planning plus four
kernels per layer. Turning it off saves about 4 ms per step. That changes
arithmetic: the canary differs from the default by MAE 0.0005 (correlation
0.999994).

The `py-spy` client hung at exit in both runs. The CP-off engine sample was
lost, so its probe uses a bounded wait; timings are unaffected. Results are in
`results/b200_host_overhead/<name>/`.
