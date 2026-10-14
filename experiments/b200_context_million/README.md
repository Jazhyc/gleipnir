# Single-request contexts to 1M on the FP8 default

Measure the current optimized serving default (vLLM 0.31, FP8 full-attention
projections, FP4 MLP/GDN projections, MXFP8 attention prefill) at concurrency 1
at every power of two from 1,024 to 1,048,576 input tokens. The user requested
this baseline before long-context optimization work and asked for explicit hardware/environment
records, because nodes drift. The archived [long-context finding](../../docs/findings/b200_long_context.md)
used the superseded 0.24 FP4-projection stack and stopped at 256K.

Hypothesis: the fitted per-chunk model from the 0.24 sweep (constant
projection/MLP/GDN cost plus attention cost linear in history) still describes
the FP8 default, and attention dominates beyond 64K. This is a systems/capacity
baseline: no labels, no long-context quality claim, no promotion.

## Contract

- Workload: the deterministic unlabelled synthetic generator from
  `b200_long_context` (`synthetic()`), exact decode/re-encode token counts.
  Prompts at 8K–256K must be byte-identical to the archived `long02` workload.
- Timing: one excluded warmup, then five sequential requests per length with
  fresh HTTP pools. Latency includes localhost HTTP, native text encoding,
  engine waiting and inference. Report median/range latency, input tokens/s,
  requests/s, peak allocated/reserved/incremental PyTorch memory, observed
  query/history attention shapes and GPU clock/power/throttle samples.
- Recipe: unchanged selected command except `--max-model-len`, the telemetry
  worker and, in the YaRN phase, RoPE scaling. 32,768-token chunk budget,
  128-sequence cap, FCFS, prefix caching off, source-bound zero-output-token
  pooling reservation, direct FROST/native Gigatoken.
- MXFP8 envelope: only the descriptor/producer length bound changes. Before any
  server starts, `native.py` requires bitwise-unchanged short/batched outputs,
  finite full 32K query chunks against 64K–1M histories placed beyond 2^31 cache
  elements, an irregular 32K+17 history and sampled independent quantized
  arithmetic agreement <=1%. Strict BF16-oracle deviations are recorded
  separately and are not a gate.

## Phases

1. `native`: checkpoint RoPE, `--max-model-len 262144`, lengths 1K–256K
   (powers of two). The real 20-row canary must reproduce the FP8 default (MAE <=0.005, correlation
   >=0.995, finite, nonzero adapter effect) — this also validates the restored
   node. Three full320 c128 passes are compared with the six archived FP8
   default repeats as a short-input batch diagnostic.
2. `yarn`: beyond the checkpoint's 262,144 positions, Qwen's documented YaRN
   (`factor` 4, original 262,144, mRoPE sections kept; vLLM 0.31 also needs the
   scaled `max_position_embeddings`), `--max-model-len 1048576`, lengths 32K/256K/512K/1M. vLLM's YaRN is static, so it changes
   scores at every length; its canary is recorded as a diagnostic (finite and
   nonzero effect required, reproduction not required). The 32K/256K overlap
   isolates any RoPE timing effect. Timing is valid; scores beyond 256K are not
   validated monitor outputs.

## Environment receipts

`environment.py` records driver/CUDA, GPU UUID/serial/VBIOS/PCIe/clocks/power
limit/ECC/MIG, CPU model and quotas, memory, kernel/OS, Runpod Pod/data-centre
IDs, key package versions, a digest of all installed distributions and the lock
hash. It snapshots at start, after each server is ready and at the end; raw
`nvidia-smi -q -x` and `lscpu` are kept. No CUDA context is created.

Stop on identity/native/canary/token/finite failure, OOM, timeout or
completion. Record each as a distinct outcome and retire only this run's
identity-verified server. Do not change capacity or restore another server.

```bash
PYTHONPATH=src:. /tmp/gleipnir-vllm031-runtime/bin/python \
  -m experiments.b200_vllm031.runtime \
  -m experiments.b200_context_million.run --name million01
```

`--phase native` or `--phase yarn` runs one phase. Export tables and figures
with `python -m experiments.b200_context_million.summarize
results/b200_context_million/million01`. The completed `million01` run reaches
1M in 43.08 s; see [the finding](../../docs/findings/b200_context_million.md). Results are under
`results/b200_context_million/<name>/`; server logs under
`logs/runpod/b200_context_million/`.
