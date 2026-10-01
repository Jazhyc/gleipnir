# Native Four Over Six MLP training on B200

Date: 2026-10-01. Contract:
[`b200_fouroversix`](../../experiments/b200_fouroversix/README.md).
Status: native kernel canaries passed; both full-model preflights failed the
same-weight compilation gate. Stopped after the requested boundary change.
No recipe promotion.

## Intervention and controls

Use the existing Runpod B200 and selected twelve-checkpoint, selectively compiled,
16,384-padded-token/max-eight adaptive physical batches, with 32 examples per
logical AdamW update. Preserve Qwen3.5-4B revision, frozen data/soft targets,
rank-128/alpha-256 FP32 adapters, SDPA and pinned FLA/causal-conv1d/Triton.
Only MLP bases change precision. Attention retains NF4 storage and BF16 compute
from the optimized recipe. This is frozen-base LoRA, not full-parameter training.

Four Over Six 1.0.5 is an isolated, source-built overlay (10m59s build), with
explicit Triton quantization and CUTLASS native NVFP4 matrix multiplication.
Use original BF16 checkpoint MLP weights, 4/6 MSE scale selection, 2D weight
blocks, nearest forward and stochastic input-gradient quantization. Prepack
frozen forward/transposed weights; keep higher-precision masters/LoRA branches.

Upstream 1.0.5 linear backward asserts physical batch one and computes base-weight
gradients even for frozen bases. The tested local wrapper flattens arbitrary
leading dimensions and computes only the frozen base's input gradient, while
using unchanged upstream native quantization/GEMM functions. No upstream package
patch or higher-precision GEMM fallback is used.

## Native kernel evidence

On NVIDIA B200/SM100, Torch 2.11.0+cu130, native arbitrary-batch forward and
backward passed against FP32 multiplication of decoded FP4 operands:

| Physical batch | Forward relative L2 | Input-gradient relative L2 |
| --- | ---: | ---: |
| 1 | 0.001663 | 0.001640 |
| 2 | 0.001645 | 0.001645 |
| 4 | 0.001661 | 0.001668 |
| 8 | 0.001662 | 0.001657 |

Stochastic backward was finite. The recorded extension is the isolated compiled
`fouroversix/_C.cpython-312-x86_64-linux-gnu.so`; explicit backend selection
prohibits simulated/reference matrix multiplication. Reference decoding is used
only to validate the native arithmetic. Evidence:
`results/b200_fouroversix/kernel_canary.json` and its canary log.

## Initial full-model compilation failure

The original global longest-32 selection loaded successfully with required
FLA and causal-conv1d bindings. Before long-context backward or an optimizer step,
the two-input same-weight eager/compiled loss gate failed: eager **1.352499**,
compiled **1.553454**, a **14.9%** increase. Keep the frozen absolute 0.01 plus
1% relative loss limit. This is not evidence of quantized training convergence,
valid native compiled execution, or a speed result.

The initial wrapper keeps native base operations opaque but permits surrounding
MLP arithmetic to compile. The failure motivates isolating that larger boundary;
the cause has not been established. Artifacts/logs under
`results/b200_fouroversix/` and `logs/runpod/b200_fouroversix/` were collected
locally before follow-up. No optimizer update occurred in the failed campaign.

## Separate eager-MLP boundary diagnostic

`eager_mlp_config.yaml` preserves all numerical gates and longest-row preflight
while keeping entire MLPs eager. Surrounding decoder layers remain selectively
compiled, with the same checkpointing and adaptive batching. The planned controls
compare the original
optimized NF4 control, an NF4/eager-MLP boundary control, BF16/eager MLP bases,
and Four Over Six/eager MLP bases. Match initial FP32 adapter hashes, cohort and
logical update membership. Ten-step training and before/after training probes
were planned as bounded diagnostics, with no held-out quality selection.

The native arithmetic canary passed again. All 32 MLP interfaces were confirmed
eager, all 96 MLP bases used the native wrapper, and FLA/causal-conv1d bindings
were confirmed. The model gate nevertheless failed with exactly the same values:
eager **1.352499008178711**, compiled **1.5534536838531494**. Expanding the eager
boundary did not resolve the disagreement; this result does not establish its
cause. No long-context backward or optimizer update occurred in either run.

The user asked to stop after trying this change. The launcher was paused while
the current preflight continued, then allowed to record its failure and exit.
No ten-step controls or further variants ran. Both failed campaigns' JSON
reports, contracts and logs were collected locally; follow-up evidence is under
`results/b200_fouroversix_eager_mlp/` and
`logs/runpod/b200_fouroversix_eager_mlp/`. The B200 Pod remains running and idle.
There is no model-training throughput, memory-saving or learning-quality result.

Focused local verification: 43 wrapper/batching/audit/launch tests passed before
the original run; 15 boundary/checkpoint/compile tests passed for the follow-up.
Ruff passed. Native isolated backward is validated; the optimized full-model
training integration remains blocked by the unchanged compilation gate.
