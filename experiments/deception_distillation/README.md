# Deception distillation

This is the first Gleipnir experiment and a cleaned port of the Aletheia's Quest
privileged-information distillation work. It tests whether a Qwen 3.5 monitor can
learn a stronger teacher's calibrated binary margins and compact evidence.

## Hypothesis

Soft literal `0|1` teacher probabilities plus compact evidence-based targets will
outperform hard-label-only LoRA training on held-out macro AUROC and calibration,
without materially increasing benign false-positive rates.

## Ported components

- completion-only Qwen LoRA SFT with optional direct, pairwise, binary-soft, and
  ordinal-soft objectives;
- prompt construction and teacher-output parsers in `gleipnir.prompts`;
- soft-target cache aggregation;
- local vLLM/Transformers training configuration;
- generic OpenRouter logprob collection through `gleipnir-openrouter`.

Historical findings are preserved in
`docs/migrations/aletheia_distillation_findings.md`; they are prior evidence, not
a license to select on the old competition test sets.

The shared trainer also supports deterministic proportional,
square-root-balanced, or uniform-dataset sampling and configurable model-only
checkpoint retention. Training metadata records expected dataset exposure,
optimizer/scheduler settings, LoRA dropout, target scaling, checkpoints, and
loss/learning-rate history.

Direct-boundary MIL also has an opt-in two-process `torchrun` launcher. Specify
`world_size: 2` and the per-rank accumulation explicitly in the job contract;
the launcher does not silently rescale batches. Selected final/MIL LM-head
projections dispatch through DDP.forward so its reducer sees every backward.
Each rank loads its own QLoRA replica on LOCAL_RANK, uses isolated compiler
caches, and verifies all trainable parameters agree exactly before completion.
Metadata records global batch and distributed/checkpointing policy. Other
custom auxiliary objectives are not yet validated for this DDP route and fail
closed. Eager operation retains the required FLA/causal-conv1d kernels.

The custom trainer returns microbatch-mean losses and explicitly disables
Transformers' inferred token-count loss normalization. This keeps direct and
sequential auxiliary gradients on the same accumulation scale; metadata records
`explicit_microbatch_mean_v1`. See the
[objective accumulation audit](../../docs/findings/monitoring_objective_accumulation.md)
for the affected historical rationale runs.

Opt-in `student.training.adaptive_microbatching` supports single-device,
dropout-free binary hard/soft training. Set `per_device_train_batch_size` to the
desired logical optimizer batch and `gradient_accumulation_steps: 1`. Trainer
splits each logical batch into length-sorted physical microbatches, bounded by
`max_padded_tokens` and a power-of-two `max_micro_batch_size`. Long inputs remain
singletons. Weight each microbatch mean by its fraction of the logical batch;
metadata records `sum_per_example_over_logical_batch_v1`, realized partitions,
padding, timing and peak memory. A partial final batch uses its actual size.
Auxiliary, dataset-reweighted, distributed, dropout and in-training evaluation
paths fail closed. See the [B200 profiling experiment](../b200_adaptive_microbatching/README.md)
for the empirical selection protocol; token budgets are not memory guarantees.

For the user-selected B200 recipe, `student.training.gated_delta_backend=flashqla`
and `student.training.gated_delta_parity_policy=selected_finite` install pinned
FlashQLA in all 24 GDN layers with BF16 operands and FP32 gates/normalization.
Use the reusable `qwen35_4b_b200_default` profile and worker so the isolated
package is exposed before import. Record all strict parity results; the selected
policy accepts finite numerical differences, and every optimizer batch rejects
missing/nonfinite gradients. BF16 FA4, causal-conv1d and FP32 master adapters remain
part of the recipe. See the [decision](../../docs/decisions/b200_flashqla_training_recipe.md).

The B200 default now installs native NVFP4 MLP forward/base input gradients with
hardware activation packing and fused descale through the normal Trainer.
The profile selects explicit finite acceptance and checksum-bound receipt reuse,
preserving failed strict loss/gradient diagnostics. FP32 adapters and BF16
GDN/FA4 remain intact. Keep the named `qwen35_4b_b200_bf16_fa4` comparison and
frozen campaign recipes. See the [current decision](../../docs/decisions/b200_native_fp4_training_recipe.md).

## Inputs

Materialize training records as JSONL outside Git. Each usable record needs a
stable `dataset`, `index`, `label`, `student_prompt`, and `student_target`. Soft
teacher files additionally key `soft_target` by the same dataset/index identity.
Record upstream dataset revisions and licenses in the experiment result manifest.

## Run

```bash
python experiments/deception_distillation/train_student_sft.py \
  --config-path . --config-name config

sbatch cluster/slurm/train_deception_distillation.sh
```

The checked-in config is a starting recipe. Replace artifact paths with a newly
materialized, source-documented dataset before launching a real run.
