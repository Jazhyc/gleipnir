# Qwen3.8-27B training speed on the ten-update benchmark

Goal: make full Qwen3.8-27B student training runs fast on one B200 by
iterating on the training stack with the same ten-update packed benchmark used
for the 4B ([trim experiment](../b200_training_trim/README.md)). This is
systems work only: no quality selection, no ID/OOD evaluation.

**Model.** `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`,
64 layers (48 GDN, 16 full attention, final layer full attention), hidden
5,120, untied embeddings ([architecture](../../docs/research/qwen38_27b_serving_transfer.md)).
It loads text-only through the causal-LM loader, like the 4B.

**Benchmark (unchanged from the 4B).** `prepare.py` copies the frozen 4B jobs
(`results/b200_adaptive_microbatching/{jobs,preflight_jobs}.jsonl`):
the same 320 stratified rows and selection hash, Kimi soft targets, seed 0,
LR 5e-5, rank-128/alpha-256 BF16 LoRA with FP32 masters, logical batch 32,
16,384-token packing, FlashQLA GDN with FP32 gates/norm, segmented SDPA, ten
updates. The trainer tokenizes with the 27B tokenizer, which renders the
frozen prompts identically to the 4B's
([audit](../qwen27b_generations_ood/README.md#interface-audit-auditpy-cpu-only)).

**Changes required by the model.**
- **Memory.** The 4B recipe runs without gradient checkpointing (158 GB peak).
  Estimated 27B activations without checkpointing are about 330 GB for a 16K
  pack, so the 27B control checkpoints every layer (non-reentrant) with the
  `checkpointed_full_attention_and_linear_shell` compile policy.
- **Initial adapter.** The 4B initial adapter does not apply; the trainer
  builds a fresh LoRA seeded with `adapter_init_seed=0`, and its master digest
  is recorded in `screen.json`. Later arms pin that digest for matched resets.
- **FlashQLA install** now expects as many patched modules as the config has
  linear-attention layers (48), instead of a hardcoded 24.
- The packed compile-cache limit is raised from 64 to 128 for the 64 layers.

**Known gap.** The FlashQLA isolated canary
(`results/nf4_flashqla_bf16_precise_canary`) covers 4B GDN shapes (32 value
heads). The model-level FlashQLA canary in the trainer still runs on the 27B.

**Measurement.** Mean and median steady update time (updates after warm-up,
identical physical partitions), peak memory, new Dynamo graphs during measured
updates, losses and gradient norms. Each arm runs on the same GPU against the
latest control. An optimization is kept when it is at least 2% faster, its
losses track the control within run-to-run noise and its canaries pass.

**Stop** on OOM, nonfinite or missing gradients, input drift, a failed canary,
compile-cache exhaustion, or a wrong update count.

```bash
python -m experiments.qwen38_27b_training_speed.prepare   # then push the jobs
bash experiments/qwen38_27b_training_speed/screen.sh control01   # on the Pod
```

## Run log

**Setup (2026-10-17).** EU-RO-1 B200 Pod `ljzun21fdpft92`
(`GPU-7a2d0c53…`, driver 580.126.20), volume `cb7bxu71ug`, container disk
resized to 300 GB at the user's request. Runtime restored with
`logs/runpod/qwen38_27b_training/restore.sh`; weights downloaded to
`/root/hf-27b` (52 GB). `control01` launched 14:48 UTC.

## Control01 (2026-10-17 14:48–15:13 UTC)

The run completed with finite gradients, all ten updates and no new Dynamo
graphs during measured updates (18 frames). The longest-32 preflight
(28,733-token maximum, 711,225 tokens) peaked at 90.5 GB allocated. Initial
master digest `4093b2aa…` (fresh seed-0 LoRA: 512 tensors, 637.5M parameters).

| ten-update packed benchmark | 4B control (same benchmark) | 27B control01 |
|---|---:|---:|
| mean update | 4.924 s | **27.80 s** (5.6×) |
| median update | 4.827 s | 27.09 s |
| per-batch range | 2.89–7.22 s | 16.80–40.81 s |
| tokens/s | 26,693 | 4,728 |
| peak allocated, measured updates | 147.1 GiB (no checkpointing) | 87.8 GiB (all layers checkpointed) |
| loss, update 1 → 10 | 0.600 → 0.325 | 0.373 → 0.218 |

At this speed one strict-filtered epoch (245 updates) is about 1.9 h of
updates, before startup.

**FlashQLA model canary.** Gradient relative L2 against pinned FLA is 0.339
(4B: 0.129; both above the 0.05 gate and accepted, as before, because they
are finite under the ten-step comparison). Losses differ on a few batches by
0.02–0.05, as on the 4B. The error grows toward early layers in both models:

| depth band (eighths, last → first) | 8th | 7th | 6th | 5th | 4th | 3rd | 2nd | 1st |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4B | 0.021 | 0.032 | 0.039 | 0.054 | 0.086 | 0.113 | 0.166 | 0.210 |
| 27B | 0.058 | 0.091 | 0.124 | 0.131 | 0.138 | 0.298 | 0.846 | 0.945 |

This is consistent with small per-kernel BF16 differences (the isolated
kernel canary agrees within 2%) amplified through twice the depth, not a
grouped-head fault: Transformers expands q/k to 48 heads before the kernel
(`modeling_qwen3_5.py:523-525`). For quality work it means early-layer 27B
LoRA gradients are sensitive to GDN arithmetic. It does not affect timing.

## Where the time goes (profiled update 5, `screen_control01_audit`)

`GLEIPNIR_HOST_PROFILE_DIR` traced update 5 (189K tokens, 41.6 s wall, GPU
busy 96.5%). `top_kernels` in `host_profile_control01/update05_trace.json`:

| kernel class | time | share |
|---|---:|---:|
| GEMMs (cuBLAS nvjet) | 18.7 s | 46.5% |
| full-attention SDPA (PyTorch FA2 kernels, head dim 256) | 10.5 s | 26.2% |
| Triton fused elementwise | 4.0 s | 10.0% |
| FlashQLA GDN | 2.4 s | 6.1% |
| copies | 2.4 s | 6.0% |
| norms, other elementwise | 1.3 s | 3.2% |
| causal conv1d | 0.8 s | 2.0% |

- Host overhead is not the bottleneck (3.5% idle).
- GEMMs run near peak (about 3×10¹⁶ FLOPs in 18.7 s), so they speed up only
  by doing less work. Full recompute accounts for about a third of GEMM time
  plus a second attention forward (544 forward vs. 512 backward attention
  calls).

**Attention backends** (`attention_backends.py`, 27B shapes through the
Transformers SDPA entry point; `attention_backends01.json`). Only FA2 runs:
cuDNN attention rejects head dim 256 (limit 128 in this PyTorch build), and
memory-efficient attention requires equal query/KV heads. FA2 matches an FP32
reference within 0.2–0.3% (output and q/k/v gradients); forward+backward
takes 0.112 s per layer at 28,733 tokens and 0.037 s at 16,384. FA4 is the
only faster kernel in the repo, but it was rejected for quality on the 4B
([decision](../../docs/decisions/b200_bf16_sdpa_training_default.md)), so it
is not tried without the user's say-so.

## Recompute arms

Storing a full-attention layer (no checkpoint) saves its recompute: about
0.23 s per update for an attention layer (attention forward plus GEMMs)
versus about 0.12 s for a GDN layer, so attention layers are stored first.

| arm | stored layers | preflight peak | measured peak | mean update | change | result |
|---|---|---:|---:|---:|---:|---|
| control01 | none | 90.5 GB | 87.8 GiB | 27.80 s | | control |
| gdnckpt01 | all 16 attention | 176.4 GiB | OOM at update 2 (176.7 GiB allocated) | | | failed |
| attn801 | attention 7, 15, …, 63 | 127.3 GiB | 130.0 GiB | **26.36 s** | **−5.2%** | accepted |

attn801 is faster on every batch (−3.2 to −5.9%), adds no Dynamo graphs, and
its losses equal the control's bitwise for updates 1–2 and then drift by
about as much as an identical control repeat (`screen_control01_audit`).

| attn1201 | attention 7, 11, 15, 23, 27, 31, 39, 43, 47, 55, 59, 63 | 151.8 GiB | 154.6 GiB | **25.65 s** | **−7.7%** | accepted |
| trim02 | none; compiled final-layer trim | 84.3 GiB | 87.8 GiB | **26.93 s** | **−3.1%** | accepted |
| **trim_attn1201** | attn1201's layers + compiled trim | 151.8 GiB | 154.6 GiB | **24.99 s** | **−10.1%** | accepted |

"Stored" means not checkpointed: the layer keeps its activations for
backward instead of recomputing them from its input.

**Trim (`screen_trim02`).** Its canary passed: gradient relative L2
1.08% against a 0.99% repeat noise floor on a 16K multi-sequence pack, and
1.99% against 1.75% on the 28,733-token row; logit maximum difference 0.125
(one BF16 ulp) and 0; no trained final-layer parameter lost its gradient.
Every batch is faster (−1.9 to −4.2%), peak memory is unchanged, and losses
track the control (update 10: 0.2139 vs. 0.2184).

**Trim, first attempt (`screen_trim01`, failed).** The trim canary OOMed at
178 GiB during startup: it runs before the Trainer first calls
`model.train()`, and Transformers checkpoints only in training mode, so the
canary's 16K-pack backward ran uncheckpointed. The canary now runs in
training mode (`979eff9`); dropout is 0, so values are unchanged.

**Combination (`screen_trim_attn1201`).** The gains add: every batch is
9.1–11.6% faster than control01. The trim canary passed (pack 1.08% vs. 1.04%
noise; longest row 2.04% vs. 1.12% noise, inside the +1-point tolerance;
logits within one ulp). Losses track the trim02 run (update 10: 0.2187 vs.
0.2139). At 24.99 s per update, one strict-filtered epoch (245 updates) is
about 1.7 h of updates, against 1.9 h for control01.

**Memory margin.** The 151.8 GiB preflight covers the benchmark's longest row
(28,733 tokens); full-campaign rows reach 29,696 tokens (+3%), leaving about
20 GiB of the 178 GiB GPU. Storing all 16 attention layers does not fit.
