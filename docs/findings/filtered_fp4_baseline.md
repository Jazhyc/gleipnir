# Filtered FP4/FA4 baseline campaign

The user authorized setup and a new optimized-training baseline on their NC2
B200 on 2026-10-12. Pod `fs334v0u4zysvn` mounts retained network volume
`ixbh81vf9c`; direct SSH was enabled through the account proxy without a restart.
GPU UUID is `GPU-480a554d-fe81-e7b9-bd55-02ca0244f4aa`, with 183,359 MiB VRAM
and driver 580.126.09. Connection metadata is in `.runpod/qat-nc2.json`.

The pinned training/serving runtime archive passed SHA256 verification and was
restored to its original paths. Dependency overlay links, native serving sources
and container caches were restored from the retained archives. The FP4 runtime
verifier reports Torch 2.11.0+cu130, Transformers 5.14.1, FA4 4.0.0b33,
Triton 3.7.1, PEFT 0.19.1 and cuDNN 9.26/frontend 1.31.0 with the expected
projection-GEMM source hash. Existing shared compiler caches are reused.

[`filtered_fp4_baseline`](../../experiments/filtered_fp4_baseline/README.md)
defines the frozen contract. Preparation binds 86 prerequisite artifacts and
removes exactly 852/8,688 rows, retaining 7,836 rows, 75,851,675 untruncated
tokens and 245 updates. Original Kimi soft targets and zero-B initialization
are unchanged. The existing filtered SDPA ID predictions are the comparison;
no redundant control training, OOD evaluation or QAT algorithm is introduced.

The detached shared campaign driver launched as PID 574 (Trainer PID 651), chaining
training, master reference, ephemeral BF16 merge, merged reference and optimized
serving/ID evaluation with required new-adapter parity. Launch and frozen input
receipts are collected locally under `results/filtered-fp4-fa4-baseline01/` and
`data/filtered-fp4-fa4-baseline01/`; stage logs remain on the volume under
`logs/runpod/filtered-fp4-fa4-baseline01/`. This launch receipt is not a completed
quality result. No in-chat scheduler is available; the Pod stays running.

Training completes all 245 updates and the exact one-epoch population. Trainer
runtime is 2,336.893 seconds (38 minutes 57 seconds), mean training loss is
0.1944368, and steady update timing averages 9.0347 seconds after the first
86.6203-second update. Initial GPU memory is about 146 GiB. The completed FP32
master SHA256 is `4eb97e5385f53b135c0b77cb92d0cbbfa78a661100f9ba8ac4205dbabb34f057`.
Master-reference scoring, merge and merged-reference finish. Master/BF16-merge
parity passes: score MAE 0.0011047, maximum difference 0.0074601 and correlation
0.99998785, with nonzero adapter effect 0.87237.

Optimized evaluation fails before producing canary predictions or ID metrics.
The MXFP8 attention kernel's first compilation raises CUTLASS
`DSLUserCodeError[TYPE_UNSTABLE_JOIN]`: `k_row` has type `None` on one path and
`Int32` on another in generated `serving_exp2_d31b2dfc...py:1319`.
The engine exits, producing HTTP 500 at the score endpoint; the client connection
errors are secondary. The loaded CUTLASS distribution reports pinned 4.8.0.
This is an execution failure, not evidence of failed optimized score parity.
The campaign preserves its failed evaluate stage and retires the scorer; no GPU
compute process remains at the check. Training is not repeated and no arithmetic
fallback or parity waiver is applied. ID baseline completion requires fixing and
validating this serving compilation failure with the same preserved adapter.
Completion metadata, passing merge parity and failure logs are collected locally.

Verification: 29 existing campaign/filtering tests pass; log is
`logs/local/filtered_fp4_baseline/tests.log`. No training implementation changed.
