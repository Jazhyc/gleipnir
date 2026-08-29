# Kimi K3 data scaling

This experiment estimates how fixed-recipe Qwen3.5-9B monitor quality changes
with Kimi K3 annotation volume. It uses the 13,149-row Phoenix 8.1 artifact:
6,573 competition rows and 6,576 rows from four Liars' Bench configurations.

## Hypothesis and scaling form

At a fixed backbone, LoRA recipe, mixture, and validation set, the residual error
for metric `M` follows a bounded power law:

```text
M(N) = M_inf - A * N^(-alpha)
```

The primary metric is macro per-dataset AUROC. Macro balanced accuracy at the
fixed `0.5` direct-logit threshold is co-primary diagnostic evidence, not a
tuned-threshold promotion metric. Pooled AUROC and balanced accuracy are also
reported so changes in large datasets cannot be mistaken for uniform transfer.

## Intervention and baselines

The intervention is the number of annotated training examples. Fractions are
`1, 2, 4, 8, 16, 32, 64, 100%`, giving 149, 256, 511, 1,060, 2,120, 4,199,
8,398, and 13,149 rows after stratified rounding. The smallest point retains at
least one example from every available dataset/label stratum.
Selections are nested within each seed and stratified by `dataset x hard label`.
Seeds `0, 1, 2` provide three independently ordered nested sweeps. Every other
training choice is fixed to the Phoenix 8.1 recipe: Qwen3.5-9B, rank-16/alpha-32
LoRA, two epochs, AdamW at `5e-5`, effective batch size 32, and direct binary-soft
BCE. Qwen3.5 is loaded through `AutoModelForImageTextToText`, with LoRA restricted
to its language model. Cached source-specific student prompts are preserved.

The zero-data base model and full-data adapters are the endpoint baselines. This
experiment does not compare model size, teacher quality, epochs, or alternative
data mixtures; those require separate interventions.

## Held-out selection and leakage rule

Evaluation is frozen to the old 822-row competition validation set spanning 21
datasets. No validation example is used for fitting, early stopping, threshold
selection, or subset construction. Preparation fails if a `(dataset, index)`
identity overlaps training. Liars' Bench contributes training supervision but
not rows to this validation set, so improvements only establish transfer to the
competition distribution. They do not establish Liars' Bench held-out scaling.

The final competition test split remains untouched. Dataset- and label-stratified
sampling controls mixture drift, but it is not a grouped causal estimate when
upstream examples share latent generator or conversation lineage.

## Stop condition

Run the predeclared 24 training jobs once. Do not add fractions, seeds, tune the
threshold, or change the recipe after inspecting validation. Stop and mark the
run invalid if input checksums change, train/validation overlap is nonzero, any
fraction is not nested, any adapter fails, or any direct score is missing. The
power-law fit is descriptive and is rejected as an extrapolator if its exponent
is non-positive or its fitted asymptote is outside `[0, 1]`.

## Prepare and run

The historical paths are defaults on the current cluster and can be overridden
with `KIMI_STUDENT_ROWS`, `KIMI_SOFT_TARGETS`, and
`COMPETITION_VALIDATION_SPLIT`.

```bash
bash experiments/data_scaling/launch.sh
```

Preparation writes an ignored provenance manifest, immutable selection files,
and rendered validation prompts. The Slurm array uses one `gpushort`
`rtx_pro_6000`, one CPU, and 32 GB per training job. A dependent job evaluates
all adapters in one persistent vLLM process and writes:

- `results/data_scaling/summary/replicates.csv`;
- `results/data_scaling/summary/scaling_curve.csv`;
- `results/data_scaling/summary/scaling_law.json`.

Raw/materialized source text, adapters, predictions, and runtime logs remain in
ignored artifact directories and must not be committed.

## Historical Lambda execution

This completed campaign ran on the former two-H100 `monitor-foundation`
instance, which is no longer active. `run_lambda.py` assigned the longest jobs
first to the lighter GPU lane, ran one adapter per GPU, and evaluated only after
both lanes finished:

```bash
python experiments/data_scaling/run_lambda.py
```

The runner is resumable through the same adapter-completion checks as Slurm and
writes `results/data_scaling/lambda_status.json` atomically for remote polling.
Its measured H100 default is microbatch 2 with 16 accumulation steps, preserving
the predeclared effective batch size 32. Under the pinned 2026-08 stack this was
slightly faster than the historical microbatch 8 / accumulation 4 layout.

## Legacy adapter correction

The first Lambda sweep trained through Transformers' Qwen3.5 causal-LM
compatibility wrapper. vLLM correctly selected the checkpoint's multimodal
conditional-generation architecture, so the causal adapter keys did not name
the wrapped language-model modules and the resulting validation scores were
invalid. Preserve those original adapters and create checksum-tracked,
multimodal-path copies before reevaluation:

```bash
python experiments/data_scaling/rebase_adapters.py
python experiments/data_scaling/evaluate.py \
  --jobs results/data_scaling/lambda_jobs_rebased.jsonl \
  --only-job seed0-f100 --smoke-rows 8
python experiments/data_scaling/evaluate.py \
  --jobs results/data_scaling/lambda_jobs_rebased.jsonl
python experiments/data_scaling/summarize.py \
  --jobs results/data_scaling/lambda_jobs_rebased.jsonl
```

The smoke test is a required gate: it fails unless the full-data adapter changes
at least one direct score relative to the base model.

## Outcome

The corrected three-seed curve rises from `0.94708 ± 0.00206` macro AUROC at
149 rows to `0.96206 ± 0.00182` at 13,149 rows; the zero-data base reaches
`0.94512`. Macro balanced accuracy rises from `0.89563 ± 0.00182` to
`0.92222 ± 0.00137`. The positive trend becomes clearest beyond 16% data. A
bounded descriptive macro-AUROC fit has exponent `0.07734` and RMSE `0.00206`,
with its asymptote at the upper bound `1.0`; retain the observed scaling result
but do not use that boundary fit for confident long-range extrapolation.

The full point table and comparison with reasoning SFT are recorded in
`docs/findings/competition_poster_evidence.md`.
