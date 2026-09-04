# Gleipnir 4B monitoring-only learning-rate screen

## Question and frozen design

Does the `5e-5` learning rate transferred from the earlier mixed-data recipe remain
best when Qwen3.5-4B is trained only on the 8,688 monitoring trajectories? This is a
one-seed pruning screen, not a final estimate.

The intervention is AdamW learning rate: `1e-5`, `2e-5`, `5e-5`, `1e-4`, or
`2e-4`. The freshly retrained `5e-5` cell is the control. Every cell uses seed 0,
one epoch, a linear schedule with 3% warmup, no weight decay, rank-128/alpha-256
QLoRA, microbatch 1 with gradient accumulation 32, and Kimi K3 soft-target BCE at
only the selected literal `0`/`1` positions. The 24 linear-attention layers are
checkpointed while the eight full-attention layers stay eager. Inductor compiles
all 32 decoder shells but leaves each complete linear-attention mixer eager. FLA
0.5.2 and causal-conv1d
1.6.2.post1 must both bind. FLA's optional backend dispatcher is disabled so the
default Triton 3.7.1 kernels are used instead of the TileLang implementation that
failed the v3 long-sequence backward compile. The two GPU lanes change scheduling
only, not the statistical design.

The original v1 recipe retained the previously proven microbatch 8 / accumulation
4 setting, but its frozen 32-longest-row preflight OOMed. Version 2's microbatch
4 / accumulation 8 fallback also OOMed. Version 3 changes only the microbatch
split to 2 / 16; effective batch size remains 32. Its TileLang preflight cleared
the memory bottleneck but hit an internal layout-inference error. Version 4 keeps
2 / 16 and freezes FLA's default Triton backend. Version 5 pins Triton 3.7.1 in
an isolated target; its matched preflight completed in 458 seconds and used about
68.6 GiB at an observed active-step snapshot. See
`docs/findings/monitoring_lr_sweep_preflight.md`.

Version 6 adopts the systems recipe selected by the subsequent matched screens:
microbatch 1 / accumulation 32, linear-attention-only checkpointing, standard
AdamW, and `full_attention_and_linear_shell` compilation. The compile policy is
20.26% faster than full-attention-only compilation on the eight-step screen and
the broader inner-mixer candidate was rejected. Version 6 repeats the
longest-trajectory preflight with a 2,048-token same-weights logit canary before
launching any sweep cell.

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
selected-recipe longest-sequence preflight, trains two serial lanes in parallel,
parity-gates the retrained control, evaluates all cells in one persistent vLLM
engine, and writes the selection under `results/monitoring_lr_sweep/summary/`.

## Result

Version 6 completed all five seed-0 cells and the 3,012-row ID evaluation at
commit `fa3d6fb`. The frozen rule selected `2e-5` over the retrained `5e-5`
control.

| Learning rate | Macro pAUROC@20 | Macro AUROC | Macro Brier | Gloom pAUROC@20 | STRIDE pAUROC@20 | Eligible |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `1e-5` | 0.865921 | 0.958562 | 0.083537 | 0.780044 | 0.951797 | No |
| **`2e-5`** | **0.871066** | 0.957643 | **0.079273** | **0.783833** | 0.958300 | **Yes** |
| `5e-5` control | 0.847144 | 0.951173 | 0.095650 | 0.731865 | 0.962424 | — |
| `1e-4` | 0.858995 | 0.953651 | 0.090651 | 0.763415 | 0.954575 | Yes |
| `2e-4` | 0.798329 | 0.937820 | 0.088241 | 0.606458 | 0.990200 | No |

Relative to the control, `2e-5` gains `0.023922` macro pAUROC@20 and
`0.006470` macro AUROC while lowering macro Brier by `0.016378`. Its source
changes are `+0.051967` on Gloom and `-0.004124` on STRIDE, within the frozen
per-source guardrail. Although `1e-5` has the highest macro AUROC, it misses the
STRIDE guardrail by `0.000626`. The `2e-4` result is strongly imbalanced across
sources and is rejected.

Each full cell took about 2.57–2.60 hours at 0.929–0.938 examples/s and peaked
at 24.86 GiB allocated. The complete two-lane campaign, preflight, parity, and
serial evaluation took 9.19 hours. Serving parity passed. This remains one-seed
ID-development evidence: retain `5e-5` as the confirmation control, replicate
`2e-5` across seeds, and do not inspect the strict OOD suite yet.
