# Single-request long-context serving

Measure the selected cached score recipe at concurrency 1 with exact input
lengths 8,192, 16,384, 32,768, 65,536, 131,072 and 262,144. The user explicitly
requested the full range including a possible 256K OOM. Synthetic text content
is acceptable. This is a systems/capacity experiment with no labels, AUROC on
synthetic text, model-quality claim, promotion or final-ID selection.

Hypothesis: long contexts expose quadratic full-attention/history-packing costs
and the actual serving capacity boundary. Freeze deterministic plain-text
prompts with exact decode/re-encode counts and tokenizer/source checksums.
At each length exclude one first-use warmup, then time five sequential requests
with fresh HTTP pools. Include native text encoding, localhost HTTP, engine
waiting and inference. Report median/range input tokens/s, requests/s, request
latency and peak PyTorch allocated/reserved GPU memory; persistent KV/state
allocation is included in absolute memory, and NVML snapshots are separate.

First measure the existing warm reference at 8K/16K/32K and save its explicit
64K HTTP rejection. Both its CLI and MXFP8 adapter have a 32K context limit.
The checkpoint natively supports 262,144 positions. Then retire that verified
server once, validate an opt-in 262K MXFP8 descriptor/guard envelope, and launch
one extended server for all six lengths. Preserve the 32,768-token chunk budget,
128-sequence engine cap, weights/head, native Gigatoken/direct FROST, FP4/MXFP8
arithmetic, BF16 cache, FP32 gates/state, FCFS and prefix caching off. The worker
changes only the MXFP8 length bound; original serving sources remain unchanged.
The new envelope is separately source/runtime/GPU bound. Require bitwise old/new
outputs at short and 128-sequence shapes, finite full-32K query chunks against
64K/128K/256K histories, sampled independent MXFP8 arithmetic agreement <=1%,
and an irregular 32K+17 history check before server launch. Record strict BF16
oracle deviations separately from quantized arithmetic acceptance.

Recheck the real 20-row adapter canary. Preserve three full320 c128 prediction
passes against all six archived selected quality controls as a batch-shape
diagnostic, not long-context quality validation. Keep startup/first-use time
separate from warmed serving timing. Stop on identity/native/canary/finite/token
failure or completion. Record OOM, HTTP rejection, timeout and worker failure as
distinct outcomes. Preserve partial measurements before any failure. Do not
automatically restore the reference after failure or change billable capacity.
Keep a successfully completed extended worker warm for compatible trials.

```bash
PYTHONPATH=<frozen-source-bootstrap>:src:. <serving-python> \
  -m experiments.b200_long_context.run --name long01
```

Push only this experiment directory to the existing B200; do not overwrite its
frozen serving sources or metadata reconciliation. Results and executed sources
are under `results/b200_long_context/<name>/`; server lifecycle uses the shared
identity-checked stop helper and metadata directory.

The first extended attempt exposes a pinned vLLM 0.24.0 boundary bug: a running
chunked pooling request reserves one generated-token position and stalls before
its final input token at the exact context cap. Preserve `long01/boundary_stall.json`
and that failed attempt. `PoolingContextScheduler` changes only this reservation
to zero, with CPU-source/runtime checks and upstream FCFS order unchanged. A CPU
test using the installed scheduler reproduces the stall and schedules the last
token after correction. Retry with `python -m experiments.b200_long_context.resume
--name long02 --failed-name long01`, reusing native/quality receipts and all GPU
sources/compiler identity, rechecking the real canary and rerunning all six
lengths on the same corrected configuration.

The corrected sweep completes exact 256K without OOM at C1, with 3.60 s median
latency and 72,795 input tokens/s. See [the finding](../../docs/findings/b200_long_context.md)
for all lengths, memory/quality diagnostics and the preserved failed attempt.
Export figures with `python -m experiments.b200_long_context.summarize
results/b200_long_context/long02`.
