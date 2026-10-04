# Native FP4 MLPs after leaving QLoRA

Date: 2026-10-04. Completed on the existing US-NC-2 B200. Native FP4 MLPs
work with the current packed LoRA recipe, but measured updates take 53.44%
more time than the matched BF16 control (7.86192 versus 5.12363 seconds).
Native arithmetic, per-token scaling, eager/compiled packing, longest-batch
memory checks and all 20 finite updates pass. Keep the BF16 MLP default.

## Intervention and control

Load Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
without bitsandbytes or k-bit preparation. Replace only the 96 frozen MLP
gate/up/down bases with Four Over Six 1.0.5 native CUTLASS W4A4 forwards.
Use strict per-token 4/6 activation selection and fused scaling. Input gradients
use the exact decoded forward weights in BF16. The other base components retain
BF16, and trainable rank-128/alpha-256 LoRA master parameters retain FP32.
Original BF16 MLP weights and decoded backward weights remain resident;
this does not establish a four-bit memory footprint or FP4 backward.

The control is the completed clean segmented-SDPA condition in
`results/b200_bf16_fa4_accepted/summary.json`: final ten updates average
5.123633625 seconds, with peak allocated memory 146.995065689 GiB. Preserve
the frozen 320-row selection, soft targets, initial master tensor digest
`a6b1d2e9fd89efff9523150a76035a2e5d27900eaae3c7a4820e3b9277078f11`,
logical batch 32, 16,384-token packing, 29,696 context, 24 FlashQLA GDN layers,
selected-token projection, optimizer and scheduler. No model checkpointing.
Require exact physical batch/token agreement before comparing measured updates.

The [experiment contract](../../experiments/b200_fp4_mlp_lora/README.md)
defines 20 updates, ten warmup, and a 5% improvement threshold for recommending
follow-up. Strict packing gradient tolerance stays 5%; the earlier explicit
FA4 learning acceptance does not authorize relaxing this FP4 gate. No teacher
calls or held-out evaluations occur in this systems screen.

Runtime retains the existing B200, Torch 2.11.0+cu130, Transformers 5.14.1,
PEFT 0.19.1, FLA/fla-core 0.5.2, FlashQLA revision
`da06429d54b0f577de0a638f451ac8f0b395e0ac` and Triton 3.7.1.
Four Over Six uses its isolated 1.0.5 native extension. Kernel and compiler
caches reuse `.cache/training/shared/gpu-0` on the persistent network volume.
Commands, source archives, input checksums, effective cache paths and receipts
are recorded with each attempt.

## Startup evidence

Fused normalization/rescaling is bitwise identical to the unfused row reference
on widths 2560 and 9216, irregular row counts and the 16,384-token shape.
Native batch-1/2/4/8 forward relative L2 against decoded reference arithmetic
is 0.002301–0.002345; decoded-BF16 backward error is 0.001647–0.001664.
Row independence error is exactly zero. These validate local arithmetic,
not agreement of the FP4 model with the original BF16 model.

The first Trainer attempt stopped before any update: unquantized loading kept
the model on CPU until Trainer placement, whereas native weight conversion
requires CUDA immediately. It is preserved under `results/b200_fp4_mlp_lora/`.
The repair moves the original BF16 base and FP32 adapters to CUDA before native
conversion. Its separate outputs are `results/b200_fp4_mlp_lora_cuda/` and
`logs/runpod/b200_fp4_mlp_lora_cuda/`.

The focused CPU suite passed 87 tests on the pod with CUDA hidden, including
mixed-precision scope, loading rejection, packing contracts, frozen workload
construction and existing BF16/FA4 behavior. Local tests were obstructed by
filesystem I/O waits during Torch import; the remote test receipt is retained.

Eager packed-versus-independent adapter-gradient relative L2 is 0.0072918674;
compiled is 0.0093558719. Both pass the unchanged 0.05 gate. All three length
pairs have zero repeat, prefix-perturbation and cross-input-gradient effects.
The adaptive accumulation comparison also passes at relative L2 0.0096193512
with gradient cosine 0.9999537406. Longest-actual-input preflight backpropagates
32 examples, maximum length 28,733 and 711,225 actual tokens, with finite
nonzero gradients and unchanged masters. Peak allocated memory during preflight
is 182,914,818,048 bytes (170.353 GiB). The control reused validation and did not
repeat that preflight; its recorded peak and the candidate's combined peak have
different scopes.

The separate original-FLA/FlashQLA comparison fails strict parity at adapter
gradient relative L2 0.7363542694. Candidate/reference losses and gradients
remain finite, so the standing selected recipe records
`accepted_for_selected_recipe=true` separately from `passed=false`.
This exceeds the BF16-only screen's 0.152627 backend discrepancy and is not
covered by the user's FA4-specific approximately 8% acceptance. It is neither
a packed-example leakage failure nor evidence of training quality equivalence.
Any speed improvement requires separate quality validation before adoption.

## Completed timing screen

Both trajectories start with the exact same FP32 master tensor digest and execute
the same physical batches, logical indices and actual/padded token counts.
The candidate changes the final master digest and invokes all 96 native modules:
23,040 native forwards, 19,000 decoded-BF16 input-gradient backwards and zero
FP4 backwards, including startup probes. The mixed-precision verifier confirms
that no bitsandbytes base remains and non-MLP storage stays BF16.

| Measurement | Clean BF16 SDPA control | FP4 MLP LoRA |
| --- | ---: | ---: |
| Final ten updates, total seconds | 51.23634 | 78.61918 |
| Final ten updates, mean seconds | 5.12363 | 7.86192 |
| All twenty update durations, seconds | 311.26767 | 165.43565 |
| Trainer loop, seconds | 316.2488 | 169.6490 |
| Recorded peak allocated memory, GiB | 146.9951 | 171.6159 |
| Training invocation, including startup, seconds | 507.4012 | 682.1388 |

FP4's lower all-update/Trainer total does not contradict slower measured updates:
the control's first update took 210.816 seconds including compilation, whereas fresh FP4
gates front-loaded compilation. Compare the frozen final-ten window for update
speed; do not infer startup savings from the loop totals. Row and native probes
cost another 25.700 and 110.806 seconds before the FP4 training invocation.
Persistent caches were reused, so these are neither cold-start measurements nor
replicated startup estimates. The FP4 recorded peak includes fresh preflight;
the control reused it, as noted above. The candidate's final peak exceeds its
170.353 GiB preflight peak.

This single screen supports operability after removing QLoRA, and rejects the
5% throughput-improvement hypothesis for this configuration. It does not identify
which quantization/scaling/launch costs explain the slowdown, establish a quality
regression, or rule out other FP4 implementations/compiler policies. There is no
reason to promote this configuration, combine it with FA4, or launch a quality
campaign on the strength of these timings. No downstream evaluation was run.

Final artifacts and runtime logs are collected locally, including FP32 master
adapters and executed source archives. Five remote/local SHA-256 checksums match
(summary, master weights, training metadata and both model-gate receipts).
The master safetensors header contains exactly 256 FP32 tensors; all 13 archived
source checksums match the executed-source receipt. Finite-gradient enforcement
is enabled during every update, and the final master differs from the initial.
Preserve both attempts. The NC2 pod remains running and shared caches persist;
no lifecycle change was requested.
