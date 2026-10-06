# Native inference kernel screening on B200

Decision date: 2026-10-06. The user requests kernel optimization following the
completed merged BF16 baseline. Use the existing NC2 B200 and the frozen small
production-development workload, measuring latency and input tokens/s. Do not
allocate new capacity or repeat a completed BF16 control.

The live vLLM 0.24.0 engine already uses FlashInfer/TRTLLM full attention and
FlashInfer GDN prefill. Its unquantized CUDA linears dispatch to PyTorch
`functional.linear`; kernel-config `linear_backend` selects quantized GEMMs.
Training FA4/FlashQLA/cuDNN-FP4 autograd wrappers are not used by this serving
engine. Use native inference integration first: it handles static weight
packing, dynamic activation scales, fused projection layouts and CUDA graphs.
Kernel superiority remains empirical for this workload.

The first intervention is MLP-only stock `fp8_per_channel` online quantization:
per-output-channel FP8 weights and per-token FP8 activations, BF16 output and
all attention/GDN/non-MLP linears BF16. This is available in the pinned install,
with no new package or custom GEMM arithmetic. A previous SM120 experiment
motivates the choice but uses a different adapter, workload and architecture
backend. Do not transfer its numerical acceptance or 1.26x gain to this run.
The installed online dispatcher supports FP8/MXFP8, but does not have the newer
online NVFP4 method documented in newer vLLM releases. Native NVFP4 checkpoint
backends exist separately; using them needs a compatible weight/scale artifact.

Keep merged BF16 API PID 72552 / engine PID 72648 resident and idle at port
8000. Start the candidate on localhost 8010 with the same memory fraction,
context/token/sequence budgets, seed, prefix-cache policy and merged ephemeral
checkpoint. The two engines fit in existing B200 memory. No concurrent timed
requests across engines. Preserve caches and original FP32 masters.

`experiments/b200_inference_kernels/README.md` freezes the hypothesis, scope,
reference, unchanged score canary, performance-interest rule and stop conditions.
Audit loaded dtypes/methods/resolved kernel classes before the canary. Fail
closed on unexpected quantization or weight-only fallback. After fresh canary
passes, run two 64-prompt passes at each client concurrency 1/4/16 and compare
with archived merged BF16 results, including paired scores, margins, flips and
within-method variation. Do not promote production precision from this small
training-seen workload or rerun full ID here. Preserve failed receipts.

Fifteen focused tests and Ruff pass before launch. Active-turn monitoring checks
startup every 30–60 seconds; no in-chat scheduler is available for later wakeups.

The first attempt rejects occupied port 8001 before starting a server or loading
weights. Preserve `fp8_mlp01/failure.json` and logs; verify port 8010 free before
the `fp8_mlp02` retry. Do not stop or modify the existing 8001 listener.
