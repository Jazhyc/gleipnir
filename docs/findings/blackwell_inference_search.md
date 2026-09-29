# RTX PRO 6000 inference search — 2026-09-29

The frozen protocol and executable conditions are in
[`experiments/fp4_inference`](../../experiments/fp4_inference/README.md).
This campaign uses the existing ordered 32-row development slice (338,780 prompt
tokens) on Slurm job 32267015, with one RTX PRO 6000 Blackwell Server Edition,
one CPU and 32 GiB host RAM. GPU work stops at 01:29:21 CEST on 2026-09-30.

## Completed master/merge reference

The original four full prompts passed the unchanged FP32-master versus merged
BF16 eager gate on this GPU. Mean/max absolute score drift was 0.005537/0.017909,
correlation 0.999073, with zero threshold flips. The maximum base-to-master
adapter effect was 0.421917, establishing that the artifact changes outputs.
This is a bounded artifact gate, not a throughput or full-split quality result.

Transformers used SDPA and its Torch gated-delta fallback because the optional
fast libraries are unavailable. This exception is confined to twelve untimed
scores (base/master/merge on four rows); serving measurements use persistent
vLLM. The frozen serving inputs, tokenizer, parent subset and merge manifest
retain their original hashes. The first import waited on shared-filesystem I/O;
this startup delay is excluded from subsequent warmed throughput measurements.

Evidence: `results/local_inference/reference.json` and
`logs/slurm/fp4_inference/reference.log`.

The first serving attempt stopped before engine construction because the
historical benchmark expects its output parent directory to exist. The new
campaign launcher now creates that parent; the failed process-timing artifact
and execution receipt are preserved as `baseline_startup_failure` artifacts.
No GPU timing or prediction was produced by that attempt.

## Completed BF16 serving baseline

One ordered pass scored all 32 rows / 338,780 prompt tokens in **9.947606 s**,
or **34,056.44 prompt tokens/s**. Source-macro AUROC was 0.921488,
pAUROC@20 0.756198, Brier 0.098918; pooled AUROC 0.909804. Gloom/STRIDE
AUROC was 0.842975/1.0, with 30 distinct scores. These are development metrics.
All prediction identities, token totals, logit margins and logprob-to-score
normalizations were independently checked against the frozen JSONL.

The serving gate passed against both eager references. Versus the FP32 master,
mean/max score error was 0.004477/0.017909, correlation 0.998958, zero flips.
Versus eager merged BF16 it was 0.001060/0.004241, correlation 0.999952.
FlashAttention 2 and Triton/FLA GDN were selected. Engine construction took
210.770 s, canary/longest-input warmup 4.080 s, whole subprocess 247.430 s.
Cold compilation took 87.90 s, followed by 41.69 s initial profiling/warmup.
Do not count these costs as warmed scoring throughput.

The scoring telemetry sample showed 100% GPU use, 58 C, 2,377 MHz and
77,519 MiB total device memory; sampled software/hardware thermal slowdown was
inactive. The 0.80 utilization policy fills memory with KV capacity, so total
device usage is not model-weight memory. Ten-second telemetry is coarse for
this short pass. Workers exited successfully. Evidence:
`results/fp4_inference/baseline/` (including `prediction_audit.json`) and
`logs/slurm/fp4_inference/20260929T195517Z-benchmark.log`.

## Kernel routing evidence

The installed locked vLLM 0.24.0 source has ModelOpt/compressed-tensors NVFP4
linear methods and FlashInfer B12X, CUTLASS and other NVFP4 kernel adapters,
but its online quantization registry has FP8/MXFP8 and no NVFP4 shorthand.
Current upstream documentation describes a newer online NVFP4 path, so local
source and executed routing govern this pinned campaign. Native SM120 support
must be established by execution before interpreting an FP4 speed result.
The [FlashInfer FP4 API](https://docs.flashinfer.ai/generated/flashinfer.gemm.mm_fp4.html)
documents backend-specific layouts and B12X selection on SM120; its online
documentation is newer than the pinned package and is advisory.
