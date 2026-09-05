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

The custom trainer returns microbatch-mean losses and explicitly disables
Transformers' inferred token-count loss normalization. This keeps direct and
sequential auxiliary gradients on the same accumulation scale; metadata records
`explicit_microbatch_mean_v1`. See the
[objective accumulation audit](../../docs/findings/monitoring_objective_accumulation.md)
for the affected historical rationale runs.

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
