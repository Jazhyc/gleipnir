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
