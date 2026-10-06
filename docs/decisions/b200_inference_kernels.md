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

Initially keep merged BF16 API PID 72552 / engine PID 72648 resident and idle at
port 8000. Start the candidate on localhost 8010 with the same memory fraction,
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

While the candidate initializes, the user questions reserving resources for an
already measured BF16 control. Stop API PID 72552 and engine PID 72648, preserving
results, logs, caches and the ephemeral merged checkpoint. Reuse the archived
control results without another resident engine or repeated baseline. The
candidate keeps its original engine budgets; no memory-fraction tuning is done.

## Completed first native kernel trial

All six FP8-MLP passes complete after loaded precision audit and fresh canary.
The audit confirms 64 native `CutlassFP8ScaledMMLinearKernel` projections with
per-channel weights/per-token activations and BF16 in every other loaded linear.
Median input throughput at client concurrency 1/4/16 rises
29,763/69,049/107,073 → 32,105/78,356/115,423 tokens/s:
7.87%/13.48%/7.80% gains including activation conversion. Interactive p50 falls
0.1041 → 0.0973 seconds; concurrency-16 p50/p95 are 0.5244/0.8681 seconds.
Only concurrency 4 clears the >10% screening interest rule; report the modest
gain rather than predicting large whole-model gains from FP8 peak throughput.

Twenty-row canary mean score difference/correlation versus master are
0.00493345/0.99966835 and versus merged BF16 0.00339731/0.99980108, passing
the unchanged limits. Nonzero effect is 0.79854948 against reused base scores.
Paired 64-row repeat medians have mean score drift about 0.0104–0.0106,
maximum 0.103560 and one changed 0.5-threshold decision at each concurrency.
Candidate within-array ranges average 0.000867, maximum 0.027824, with no
threshold-unstable rows. Passing this canary is not broad held-out quality
equivalence; do not replace a quality-validated production precision recipe.

Readiness takes 698.666 seconds, canary 10.622 and workload warmup 0.452,
excluded from timed passes. Weight memory falls 7.99 → 5.96 GiB, while total
resident GPU allocation stays 49,798 MiB under the unchanged memory budget.
Only FP8 API PID 73262 / engine PID 73329 remains warm at localhost 8010;
the old control is stopped. Collect results, executed sources, numerical
comparisons, audit and logs locally; retain disposable merged weights on the
pod and original FP32 masters/caches persistently. Fifteen focused tests and
Ruff pass. Preserve the port-collision failure, source provenance and stop receipt.
