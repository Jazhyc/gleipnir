# Gleipnir 4B monitoring-only learning-rate screen

## Question and frozen design

Does the `5e-5` learning rate transferred from the earlier mixed-data recipe remain
best when Qwen3.5-4B is trained only on the 8,688 monitoring trajectories? This is a
one-seed pruning screen, not a final estimate.

The intervention is AdamW learning rate: `1e-5`, `2e-5`, `5e-5`, `1e-4`, or
`2e-4`. The freshly retrained `5e-5` cell is the control. Every cell uses seed 0,
one epoch, a linear schedule with 3% warmup, no weight decay, rank-128/alpha-256
QLoRA, microbatch 4 with gradient accumulation 8, and Kimi K3 soft-target BCE at
only the selected literal `0`/`1` positions. FLA 0.5.2 and causal-conv1d
1.6.2.post1 must both bind. The two GPU lanes change scheduling only, not the
statistical design.

The original v1 recipe retained the previously proven microbatch 8 / accumulation
4 setting, but its frozen 32-longest-row preflight OOMed at 78.09 GiB allocated
while requesting another 3.54 GiB. Version 2 changes only the microbatch split;
effective batch size remains 32. See
`docs/findings/monitoring_lr_sweep_preflight.md`.

Training contains no deception rows. It uses the frozen five-source monitoring
artifact with 10 source-label strata and explicit lineage metadata. Preparation
checks that its trajectory hashes do not overlap the frozen 3,012-row held-out ID
development suite. The ID suite contains Gloom and STRIDE task families, so this
screen can select for those in-distribution families; the strict six-source OOD
suite remains untouched.

## Selection and stopping

The primary metric is unweighted two-source macro normalized pAUROC@20. A candidate
replaces `5e-5` only if it gains at least 0.005, loses no more than 0.01 pAUROC@20
on either source, and regresses macro Brier by no more than 0.005. Ties are broken
by macro AUROC and then lower macro Brier. The winner must be confirmed across
seeds before any final OOD evaluation or promotion.

Stop on any input checksum or recipe drift, non-monitoring training row, missing
kernel binding, longest-sequence preflight OOM, incomplete adapter parity, or
missing/non-finite ID score. The complete machine-readable contract is in
`config.json`; `id_benchmark.json` freezes evaluation.

## Run

```bash
python -m experiments.monitoring_lr_sweep.prepare
GLEIPNIR_COMMIT=$(git rev-parse HEAD) \
  python -m experiments.monitoring_lr_sweep.run_lambda
```

The launcher validates and relocates the manifests, performs the rank-128
microbatch-4 longest-sequence preflight, trains two serial lanes in parallel,
parity-gates the retrained control, evaluates all cells in one persistent vLLM
engine, and writes the selection under `results/monitoring_lr_sweep/summary/`.
