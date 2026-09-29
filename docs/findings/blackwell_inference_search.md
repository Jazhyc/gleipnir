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
`logs/slurm/fp4_inference/reference.log`. The BF16 vLLM reference is running;
no Blackwell serving-throughput result exists yet.

The first serving attempt stopped before engine construction because the
historical benchmark expects its output parent directory to exist. The new
campaign launcher now creates that parent; the failed process-timing artifact
and execution receipt are preserved as `baseline_startup_failure` artifacts.
No GPU timing or prediction was produced by that attempt.

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
