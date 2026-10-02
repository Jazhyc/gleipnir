# FlashQLA for the default B200 LoRA recipe

Status: selected by explicit user instruction, 2026-10-02.

Subsequent user selection: the current B200 default uses packed unquantized BF16
LoRA without model checkpointing. See the
[packed BF16 decision](b200_packed_bf16_training_recipe.md). The recipe below
preserves its historical selection.

## Decision

Use `systems_screen@_global_: qwen35_4b_b200_default` for future single-B200
Qwen3.5-4B NF4/BF16 LoRA training. It inherits the selected adaptive batching
recipe and enables pinned FlashQLA uniformly in all 24 gated-delta linear-attention
layers. The eight full softmax-attention layers retain SDPA; causal-conv1d remains
pinned and required. FLA 0.5.2 remains required for FP32 Q/K normalization and the
startup reference. This selection applies to B200 SM100; H100 recipes retain FLA.

Keep rank-128/alpha-256 FP32 master adapters, NF4 double quantization with BF16
linear compute, the current BF16 head, twelve checkpointed layers, logical batch
32, adaptive 16,384 padded-token/max-eight physical batches, intact oversized
singletons, the 29,696-token cap, AdamW 5e-5, norm-one clipping, linear scheduling
and 3% warmup. Compile full attention and the linear shell using the existing
Inductor policy. Each future campaign freezes its own data and training duration.

FlashQLA uses revision `da06429d54b0f577de0a638f451ac8f0b395e0ac`, automatic
partitioning disabled, BF16 Q/K/V, FP32 beta/g gates and external pinned FP32
Q/K normalization. Preserve the kernel's forward checkpoint cache. Verify the
isolated install, dependency versions, source hashes, optional compiler patches
and device capability; reject failures rather than substituting another backend.
Use `experiments/fp4_stability/bootstrap_flashqla.sh` to provision the isolated
install on a new Runpod workspace. The worker exposes it in the child environment.

## Numerical policy and evidence

The user selected this recipe after the matched ten-step comparison: FlashQLA
averages 9.0667 seconds per optimizer step versus original FLA 10.4202, a 12.99%
time reduction. Mean training losses are 0.688022 and 0.688321. Common original-FLA
probe loss starts at 1.246726 and ends at 0.912151 and 0.926935. All ten paired
FlashQLA batches are faster, with zero new compiler graphs during either measured
pass. Both runs retain verified FP32 masters. Full provenance and limitations:
[ten-step findings](../findings/fp4_training_stability.md#completed-uniform-flashqla-ten-step-learningtiming-comparison).

This is an explicit recipe selection from bounded training evidence. Strict
whole-model gradient parity still fails (relative L2 0.195302), as does the
FlashQLA eager/compiled loss gate. Preserve these failed receipts and thresholds.
Ordinary selected-recipe training records reference/candidate losses and all
master-gradient comparisons before updates. Its explicit `selected_finite` policy
accepts finite nonzero gradients despite strict numerical disagreement, and also
records finite compile and cross-microbatch canaries separately from their strict
pass/fail results. Every adaptive optimizer batch checks for missing/nonfinite
master gradients before Trainer can update. Do not apply this acceptance policy
to original-FLA profiles or frozen historical experiment contracts.

Long-run stability and held-out quality equivalence are unmeasured. Future
campaigns still freeze their stop conditions, check their actual rank/context/data
memory envelope, and select checkpoints on the agreed CoT-removed ID validation.
This decision changes the execution recipe; it does not select a model checkpoint.

## Reproduction and validation

`qwen35_4b_b200_adaptive` retains the preceding original-FLA recipe for matched
controls; `qwen35_4b_b200_fast` and all frozen experiments retain their historical
settings. The new default profile forwards its backend/policy into ordinary
training and checks the actual backend, all 24 layers and precision boundary in
training metadata. Existing ten-step diagnostics keep their bounded acceptance.
Configuration and CPU tests cover profile composition, command/environment
forwarding, strict-result preservation, finite/nonzero model gradients, rollback
on failed canaries and failure before optimizer updates. The earlier completed
GPU benchmark supplies the training evidence; this selection starts no campaign.

Validation completed: 136 focused CPU tests passed, six unsupported diagnostic
combinations skipped; Ruff and `git diff --check` passed.
