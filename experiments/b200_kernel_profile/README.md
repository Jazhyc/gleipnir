# Kernel profile of the adopted last-query default

Diagnostic only. The goal is to find where GPU time goes in the adopted FP8 default with
last-query final-layer attention, to choose the next optimization. The user
asked for kernel profiling after adopting `trim01`. No timing claims, recipe
changes or promotion follow from traced runs: profiler overhead perturbs
latency.

## Contract

Launch the selected default through the long-context runner's `LAST` variant
(the same worker as `trim01`), reusing the `million01` MXFP8 envelope receipt,
the `trim02` kernel receipt, the `million01` synthetic workload and the
existing persistent caches. The native-RoPE server profiles C1 requests at 1K,
8K, 32K and 256K, plus one full320 pass at c128. The YaRN server profiles one
1M request. Each case has one excluded warmup, then one traced pass through
vLLM's torch profiler (`/start_profile`, `/stop_profile`, frontend ignored).
The canary gates the native server, as in the sweep.

`analyze.py` reports, for each trace:

- the GPU window and busy union, and the idle share (host and launch gaps);
- kernel time by category: MXFP8 attention core, MXFP8 packing, GDN/conv,
  FP4 GEMM, FP8 GEMM, norm/RoPE, memory and elementwise;
- the top kernels by total time.

Categories come from ordered kernel-name patterns, refined after inspecting
actual names. Rerun `python -m experiments.b200_kernel_profile.analyze
results/b200_kernel_profile/<name>` after changing them. Traces are kept with
their SHA-256.

```bash
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_kernel_profile.profile --name profile01
```

`--phase native` or `--phase yarn` profiles one server. Results go under
`results/b200_kernel_profile/<name>/` and logs under
`logs/runpod/b200_kernel_profile/`. The completed traces are `profile02`
(native) and `profile03` (1M); see [the finding](../../docs/findings/b200_kernel_profile.md).

`saturation.py` (`--name saturation01`) traces the adopted default at the 2K
c32 and full320 c128 plateaus. It pairs each traced pass with an untraced
step-timed pass, and splits GPU idle into step-start input preparation, other
in-step gaps and between-step gaps. It uses the step-timing worker from
`experiments/b200_host_overhead`, which now records each step's start time.
