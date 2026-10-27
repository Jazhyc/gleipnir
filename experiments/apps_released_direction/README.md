# Injection-direction projection on the released 4B

Frozen 2026-10-21, before any score. User-requested follow-up to the
[released-monitor APPS run](../apps_released_monitors/README.md): does removing
an injection direction still suppress monitor-directed injection alarms on the
released Gleipnir 4B v0.4?

## Background

On the augmented SDPA 4B, projecting out its own layer-20 injected-minus-neutral
direction cut injected-honest FPR from 50.8% to 0.4% and raised matched AUROC
from 0.80 to 0.89, at a large injected-backdoor recall cost
([finding](../../docs/findings/b200_injection_direction.md)). Refits on the
activation-filtered and CAFT models helped much less, and the original vector
did nothing on CAFT ([CAFT finding](../../docs/findings/concept_ablation_training.md)).
The released 4B (QAD of the three-seed strict-filtered average) has never been
probed.

## Design

- **Model and serving.** The released 4B's merged BF16 checkpoint
  (`/tmp/gleipnir-merged/qad-v02-optimized01`, file hashes from its merged
  artifact), served unquantized and eager with the Lens extension. The command
  is the project's BF16 Qwen3.5-4B command (`caft-regular-sdpa02`) with only
  the checkpoint path changed. Lens 1.3.0 is installed from the hash-locked
  `b200_vllm_lens/requirements.txt` into an overlay on the image runtime. This
  is the established BF16 eager mechanistic-analysis exception.
- **Gate.** Before any APPS row: all 3,012 ID rows must reproduce the
  recorded BF16 scores (`qad-v02-bf16-01`; MAE ≤ 0.02, correlation ≥ 0.99),
  the native BF16 audit must pass, and the operator smoke (exact no-op,
  projection geometry, answer-mass readout) must pass. There is no FP32
  master on this Pod; the recorded BF16 rows stand in for it.
- **Data.** The frozen `sdpa_injection_direction01` split: 64 fit tasks / 896
  rows / 384 injected-neutral pairs; 591 disjoint test tasks / 8,218 rows. All
  9,114 prompts are byte-identical to the released 4B's APPS prompts.
- **Fit.** Unchanged fitter: unit layer-20 mean of injected minus neutral
  decision residuals over both labels.
- **Arms** (all-layer, all-token, beta 1, zero centers; same engine):
  1. plain;
  2. refit projection;
  3. cached original projection, the exact augmented-model edit
     (`directions.npz` `8c3c6e50…`);
  4. random unit projection (seed 0).
  No layer, beta or template sweep; no clamp or additive arms; nothing is
  promoted.

## Metrics

`apps_report` per arm: clean-only threshold (at most 2% clean FPR) and the
fixed plain threshold; per-template injected-honest FPR, injected-backdoor
recall, matched AUROC / raw pAUROC@20; vanilla and neutral controls; mean score
shift; answer mass on the 128-task diagnostic subset; refit/original cosine
and norms; eager-BF16 plain minus the optimized released scores
(`apps_released_monitors/apps01`). Comparison rows are the cached
augmented-model results. One checkpoint, previously studied tasks and
templates; intervals are not computed.

## Stop conditions

Input or model hash drift, fit/test leakage, failed ID gate, native audit or
operator smoke, non-finite scores or captures, Lens request-state leaks,
missing coverage, server exit or OOM. Failures are kept in `failure.json`.

## Commands

```bash
# Pod (image 0.2, after apps_released_monitors finishes):
bash experiments/apps_released_direction/pod_run.sh
```

Results: `results/apps_released_direction/released_direction01/`; logs:
`logs/runpod/apps_released_direction/`.
