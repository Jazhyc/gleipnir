# Blackwell vLLM inference search

## Frozen protocol (2026-09-29)

Hypothesis: native Blackwell low-precision linear kernels and better prefill
scheduling can improve warmed Gleipnir-4B throughput without unacceptable score
drift or runtime failures. Alternate scheduler/compiler tuning with exploration
of FP8, NVFP4 linear backends, attention, and cache precision. Record failed
conditions and distinguish numerical parity from diagnostic quality measurements.

Use the existing ordered 32-row development slice, 338,780 prompt tokens,
SHA-256 `aadb48556b134150e43946ba39d31512498e2d61b8a88a7a24fd9617b5633f03`.
Keep prompts, token counts, one-token constrained 0/1 scoring, teacher-free
labels, and released adapter/base revisions unchanged. Parent 512-row SHA-256:
`f5800ce52b38184bf3854fbf8e7a91257a3774c2f0a859c598c97e26f5829ebf`.
The development slice guides systems selection; neither it nor kernel
reconstruction error establishes population quality. Calibration, if needed,
must exclude these 32 identities and the original serving canaries.

First run the existing bounded eager base/FP32-master/BF16-merge canary on this
GPU, requiring a nonzero adapter effect and the original merge tolerances.
Then establish the merged BF16 vLLM reference with the historical two-sequence,
2,048-token budget and all original serving gates. No historical RTX 4080 speed
is used as the Blackwell baseline. Each serving process keeps one engine alive
across all canaries and the complete ordered scoring pass. Engine changes need
separate processes; batch compatible kernel measurements in one process.

For each candidate record input/model/config/code hashes, software, resolved
backend, initialization/warmup/whole-process times, warmed tokens/s, GPU
temperature/clocks/memory, raw decision logprobs, paired score/margin drift,
threshold flips, ties, source/macro AUROC, pAUROC@20, Brier and diagnostics.
Initially use one timed pass per condition, following the existing protocol;
confirm selected finalists with matched repeats only if time permits. Require
>10% warmed speed gain for interest. Select a deployable systems candidate only
if serving parity passes and development macro AUROC loses at most 0.01 and
Brier increases at most 0.01 against the new baseline. These predeclared small
development-set bounds are screening criteria, not statistical equivalence.

Quantization candidates may complete a **diagnostic** pass after a finite but
failed numerical canary under this user-authorized search. Preserve the original
limits, failed status and an explicit diagnostic reason; do not relabel failure
or promote that candidate. Stop on nonfinite/missing output, OOM, truncation,
wrong identities, compilation/backend failure or a failed baseline gate.
Record reconstruction checks before using a new FP4 artifact or custom kernel.

Compute: Slurm allocation `32267015`, `gpushort`, `roodborst3`, one RTX PRO
6000 Blackwell Server Edition (97,887 MiB visible, SM120), one CPU, 32 GiB RAM.
Driver 610.57.04; locked Torch 2.11.0, Transformers 5.14.1, vLLM 0.24.0,
FlashInfer 0.6.12. Build concurrency and ordinary CPU thread pools are one.
The existing bounded eager reference sets eight Torch threads; this remains
confined to its untimed four-row check on the one allocated CPU.

Allocation expiry is 2026-09-30 01:39:21 CEST. Stop all campaign GPU workers by
01:29:21 CEST (2026-09-29 23:29:21 UTC), leaving ten minutes. No new allocation
or extension is authorized. Monitoring uses the active agent session; no
scheduling tool is available for agent wakeups after the turn ends. Inspect
startup every 30–60 seconds and each completed condition before choosing the next.

## Run

Inside the allocation, after activating the locked environment and CUDA 13.2:

```bash
python -m experiments.fp4_inference.run --condition baseline
```

Configurations live in `configs/`. Durable logs are under
`logs/slurm/fp4_inference/`, measurements under `results/fp4_inference/`.
Completed or failed output directories are preserved. Every completed condition
gets a finding here and in `docs/findings/blackwell_inference_search.md`.

## Completed artifact reference

The four-row eager master/merge gate passed: mean/max score drift
0.005537/0.017909, correlation 0.999073 and zero threshold flips; maximum
base-to-master adapter effect 0.421917. The bounded reference used the Torch
gated-delta fallback and SDPA. No serving throughput is established by this
artifact check. Evidence: `results/local_inference/reference.json` and the
reference log. The new BF16 serving baseline has started.
