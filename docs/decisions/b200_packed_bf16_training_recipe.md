# Packed BF16 LoRA as the B200 training default

Status: selected by explicit user instruction, 2026-10-04: packed BF16 LoRA
with native FlashAttention 4 full attention. This updates the 2026-10-02 SDPA
selection. After reviewing the completed FA4 and FP4 screens, the user said to
make the recipe using FA4 standard from now on. That instruction supersedes the
screen's prospective replication prerequisite; it does not create replication
or held-out quality evidence.

The [matched FA4 screen](../findings/b200_bf16_fa4.md) observes 20.24% shorter
measured update time (5.12363 to 4.08648 seconds), with identical physical batches
and tokens. Eager/compiled packing gradient relative L2 is 0.0828666/0.0517730:
the strict 5% results remain false, with separate user-authorized 10% acceptance.
All leakage checks and longest-batch memory preflight pass; all twenty updates
have finite gradients. Keep BF16 MLPs: [native FP4 MLP LoRA](../findings/b200_fp4_mlp_lora.md)
works but takes 53.44% more measured time than BF16/SDPA.

The default profile pins packed attention to `flash_attention_4` version
`4.0.0b33`. Transformers' loader routing key stays `sdpa`; the scoped packing
router uses native causal variable-length FA4 for the eight full-attention
layers. Unpacked diagnostic calls retain the independent SDPA reference.
Reuse the completed FA4 training receipt at
`results/b200_bf16_fa4_accepted/flash_attention_4/causal_adapter/training_metadata.json`,
SHA-256 `185fa498f8ec31f07ae58a8213584c7a7f5b41738393ae75d97f91229a993364`.
Receipt reuse checks the backend, version, tolerance, original gate contents,
model/revision, BF16 loading, checkpointing, batch/context envelope and validated
B200 software versions. Raw metadata marks checks as reused, with the reference
checksum and original strict failures; it must not claim fresh passes.

The isolated `.cache/kernels/fa4` overlay goes after pinned FlashQLA/FLA and
TVM FFI dependencies on `PYTHONPATH`. Runtime retains Torch 2.11.0, Transformers
5.14.1, CuTe/CUTLASS DSL 4.8.0 CUDA-13 wheels and TVM FFI 0.1.11. Reuse
`.cache/training/shared/gpu-0` and `.cache/training/fa4_4.0.0b33_cute` on the
persistent volume. Version/hardware changes require fresh validation. No eager
fallback, cold-cache namespace or new dependency installation is implied.

Startup validation update, 2026-10-02: the user explicitly requested reusing
validation for an unchanged recipe instead of repeating diagnostic gates on
every run. Reuse the recorded kernel/packing/partition/memory receipts and mark
checks as not performed in the new run, with reference identity and checksum.
The fresh-validation procedure below applies when a recipe or its hardware,
kernels, precision, batching, packing or supported context envelope materially
changes, after relevant failures, or on request. Supported binary loss mixtures
and LR sweeps can reuse validation. Keep finite/missing-gradient checks during
updates and adapter-specific serving parity. See `AGENTS.md` and
`experiments/monitoring_hard_labels/README.md`.

Historical selection below: on 2026-10-05 the user selected
[native FP4 MLPs](b200_native_fp4_training_recipe.md) as the new default.
This BF16 recipe remains available as `qwen35_4b_b200_bf16_fa4`.

Use `systems_screen@_global_: qwen35_4b_b200_bf16_fa4` to reproduce single-B200
Qwen3.5-4B binary monitoring training. The user selected the fastest completed
packed recipe after reviewing the checkpointing/batch-size comparison. This
updates the execution default without launching a full training campaign or
changing its data, targets, duration or held-out checkpoint-selection rule.

The profile selects a frozen, unquantized BF16 causal-LM base and FP32
rank-128/alpha-256 master adapters, with model gradient checkpointing disabled.
Keep logical batches of 32 examples, accumulation one, and best-fit packing
within each logical batch at 16,384 input tokens per physical row. Oversized
examples remain intact singletons under the existing 29,696-token context cap.
Losses retain equal example weighting, including a final incomplete batch.
Retain AdamW 5e-5, norm-one clipping, linear scheduling and 3% warmup. Campaigns
still explicitly choose training duration and input/teacher identities.

Every convolution receives `seq_idx`; every FlashQLA scan receives cumulative
sequence lengths; positions reset per example. Full attention uses isolated
causal FA4 variable-length calls without a quadratic long-context mask. All 24
linear layers use pinned FlashQLA revision
`da06429d54b0f577de0a638f451ac8f0b395e0ac`, automatic partitioning disabled,
BF16 Q/K/V and FP32 gates/QK normalization. Preserve its kernel checkpoint cache.
FLA/fla-core 0.5.2, causal-conv1d 1.6.2.post1 and Triton 3.7.1 remain required.

Keep the tested selective Inductor policy `full_attention_and_linear_shell`,
precision-cast emulation, cuBLASLt and disabled BF16 reduced-precision/split-K
reductions. Use compiler recompile limit 64 and fail on exhaustion. Runtime
settings and the attention router are scoped and restored on success/failure.
The cache limit controls dispatch; it is not an accuracy setting.

For fresh validation, verify BF16 bases/FP32 masters before updates, run eager and compiled
packing isolation/parity canaries, and backpropagate the longest actual training
batch without changing adapters. Packing canaries retain the strict 5% gradient
gate and fixed leakage tolerances, with FA4's separate bounded 10% gradient
acceptance. Selected finite acceptance never overrides isolation or loss gates.
Existing FlashQLA-versus-FLA and general compiler/partition
canaries preserve strict results and the already-selected `selected_finite`
policy. Every logical update checks missing/nonfinite master gradients.
Unsupported objectives, multiple devices, incompatible kernels, leakage,
nonfinite values or OOM fail before an optimizer update; never shrink the recipe
or substitute another backend to conceal a failure.

Evidence: the matched ten-step packed BF16 screen averaged 5.1452 seconds/update
without checkpointing at 147.14 GiB peak allocated memory, versus 5.7163 seconds
with checkpointing. Increasing the checkpointed token budget to 32,768 averaged
5.5162 seconds at 111.04 GiB. All packing isolation gates passed and all measured
training values were finite. See the [packing finding](../findings/bf16_sequence_packing.md).
These are bounded systems results; longer convergence/held-out quality remain
unmeasured. Planning estimate for the current 21,837-row one-epoch corpus is
683 updates, about 47 minutes of FA4 steady compute; allow additional setup/saving
and full-population length variation.

Historical original-FLA profiles remain unchanged. The preceding packed BF16/SDPA
profile is preserved as `qwen35_4b_b200_packed_sdpa`. The preceding NF4/FlashQLA
default is available as `qwen35_4b_b200_flashqla`. Frozen JSON jobs and historical
paired packing benchmarks retain their original recipes. H100 defaults retain
their documented QLoRA/FLA recipe.

Ordinary Trainer integration is validated with focused CPU tests for scoped
settings, per-example gradients, arbitrary packed sizes and partial final
batches. The two-step B200 smoke in
`experiments/monitoring_sequence_packing/default_recipe_smoke.yaml` checks the
normal training path against the same frozen initial adapter and 320-row cohort,
including native eager/compiled isolation gates, longest-row memory preflight,
finite gradients, zero padding and a changed FP32 master. It completed both
ordinary Trainer updates, logical batches `[32,32]`, 16 physical calls (including
14-example packs), no checkpointed layers and 145.95 GiB peak allocated memory.
Its longest actual input was 28,733 tokens. Eager/compiled packing gradient
relative L2 was 0.00888195/0.00697903, with zero measured leakage. The broader
eight-example singleton/packed gradient comparison differed by 21.1489%, while
mean losses were 1.064834/1.074136. That strict 5% result remains false and is
accepted only by the existing `selected_finite` partition policy; the stricter
packing isolation gates remain independent and mandatory. Do not infer general
gradient agreement from the smaller packing probe or two successful updates.
The smoke produced 24 Dynamo graphs without cache-exhaustion fallback. It
validates the ordinary training wiring, not warmed performance or convergence.
Validation: 196 focused CPU tests passed (eight unsupported diagnostics skipped);
48 affected CPU checks passed again after the metadata update. Ruff and staged
diff checks passed. Reports and the FP32 adapter are collected locally under
`results/bf16_packed_default_smoke/`.

The complete ordinary-training metadata contract passed validation, including
14-example packs exceeding the legacy padded batch-size cap. The saved adapter
contains 256 FP32 LoRA tensors; its collected checksum matches the B200 original.
Reports, configuration, logs and weights are locally checksummed.

The FA4 default integration smoke uses two ordinary updates from the frozen
initial adapter/cohort and reuses the completed FA4 receipt. It checks launcher
wiring, metadata, finite updates and FP32 master changes; it does not repeat
numerical or memory probes or establish longer training quality. Its entrypoint
is `experiments.b200_bf16_fa4.default_recipe_smoke`, with outputs under
`results/b200_fa4_default_smoke/` and logs under `logs/runpod/b200_fa4_default/`.

Completed 2026-10-04: 92 focused CPU tests pass. The FA4 default smoke completes
both updates, 16 physical calls, zero padding, no checkpointed layers and a
changed FP32 master. Its peak allocation is 135.042 GiB; eight Dynamo graphs
are recorded. Kernel/packing/partition/memory probes are correctly marked
reused (`performed_this_run=false`), while hardware/software provenance is
verified in this run. The original strict packing failures remain in the
referenced receipts. First-update compilation is still required in a new process;
this two-step integration check makes no warmed-throughput claim.
The composed profile also passes the shared metadata validator against the actual
reused-check receipts. Artifacts/logs are collected locally; three remote/local
SHA-256 checksums match, all eight executed-source archives match their hashes,
and the saved master contains exactly 256 FP32 tensors.
