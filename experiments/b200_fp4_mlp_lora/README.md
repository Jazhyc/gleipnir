# Native FP4 MLPs with BF16 attention and FP32 LoRA on B200

Hypothesis: removing NF4 storage and k-bit preparation from the non-MLP base
allows the previously validated per-token native FP4 MLP forward / decoded-BF16
input-gradient path to work with the current packed LoRA recipe and reduce
measured update time. Quantization may change the learned function; short systems
checks do not establish held-out quality equivalence.

Use the running NC2 B200 and persistent compiler/kernel caches. Load the original
Qwen3.5-4B checkpoint without bitsandbytes, then replace only the 96 frozen MLP
gate/up/down bases with pinned Four Over Six 1.0.5 native CUTLASS W4A4. Preserve
BF16 attention, embeddings, head and normalization, FP32 rank-128/alpha-256
adapters, 24 FlashQLA layers, segmented causal SDPA, no model checkpointing,
logical batch 32, 16,384-token packing, 29,696 context and selected-token loss.
Per-token fused scaling and the strict 4/6 selector retain the preferred prior
FP4 arithmetic. Base input gradients use the exact decoded forward weights in
BF16; this screen does not claim FP4 backward or four-bit activation storage.
Reference BF16 MLP weights and the decoded backward cache remain resident.

Baseline: the completed clean SDPA condition in `b200_bf16_fa4_accepted`, with
the same frozen 320 rows, soft targets, permutation, optimizer, scheduler and
initial master digest. Candidate: 20 ordinary Trainer updates, first ten warmup
and final ten measured. Require matching physical partitions/token counts and
at least 5% lower measured total time before recommending a follow-up. Keep
every update duration, startup/check costs and peak memory. Reverse-order
replication and separate quality validation precede any default change.

There are no teacher requests, held-out selection, release or final-test access.
Fresh native arithmetic/row independence, eager/compiled packing isolation and
parity, compiler checks and longest actual batch memory preflight precede any
optimizer update. Packing's strict 5% gradient limit remains unchanged; the prior
FA4 learning acceptance is scoped to FA4. Preserve historical strict failures
and the selected finite FlashQLA policy independently. Stop on input/master drift,
kernel/fallback/version failure, leakage, failed gates, missing/nonfinite gradients,
OOM, or compiler limit exhaustion. Do not silently shrink the recipe.

First implementation keeps current compiler boundaries. If a bounded failure
diagnostic identifies surrounding normalization/activation rounding differences,
record a separate fence variant and matched BF16 fence control before timing.
Preserve failed outputs and source hashes rather than overwriting a run.

The initial attempt stopped before updates because the unquantized loader left
weights on CPU during native conversion. Preserve its receipts under
`results/b200_fp4_mlp_lora/`. The repaired path explicitly moves the original
BF16 model and FP32 adapters to CUDA before native conversion.

Current artifacts: `results/b200_fp4_mlp_lora_cuda/`; logs:
`logs/runpod/b200_fp4_mlp_lora_cuda/`. No agent heartbeat scheduler is available in
this session; startup/progress checks occur during the active turn only.
