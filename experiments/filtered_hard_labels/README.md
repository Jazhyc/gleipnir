# Hard labels on the strict-filtered recipe

The [hard-label screen](../../docs/findings/monitoring_hard_labels.md) mixed
source-label cross-entropy into the Kimi K3 soft loss with fraction 0.3. It
gained 2.8 ID pAUROC@20 points but only 0.55 OOD points, with a SHADE Arena
regression. That run differs from the current recipe in three ways:
- unfiltered data: 8,688 rows, including confident teacher–label conflicts;
- LR 2e-5;
- a soft-only control at 0.860 ID.

The current recipe applies the strict filter instead: it removes the 852
confident conflicts, leaving 7,836 rows. It uses LR 5e-5 and scores
0.896–0.914 ID across seeds 0–2.

**Question.** After filtering, the label agrees with the teacher's side on all
but 28 rows. The mixture (1 − f)·BCE(p) + f·CE(y) is then exactly BCE against
the target (1 − f)·p + f·y, which is a milder form of the sharpening that lost
6.9 points in the [temperature screen](../distillation_temperature/README.md):

| | 0.3 mix | sharpening ×2 |
|---|---:|---:|
| mean change in target | 0.014 | 0.028 |
| rows moved > 0.05 | 8.6% | 20% |

If the old gain came from correcting conflicts, filtering already captures it
and this arm should be flat or worse, mainly on Gloom.

## Design (frozen before training)

**Hypothesis.** A 0.3 hard fraction still improves held-out ID macro
pAUROC@20 over the seed-matched soft-only control.

**Arm.** `filtered-hard030-01` copies `teacher-seed-sdpa-s1-01` (seed 1,
BF16/SDPA, 245 updates, LR 5e-5) and changes only the loss weights:
`soft_loss_weight` 0.7 and `direct_loss_weight` 0.3. Identity, GPU, merge slot
and serving hash change as usual.

**Selection.** This uses the
[temperature screen's rule](../distillation_temperature/README.md#design-frozen-before-training).
Measured against the seed-1 control on the 3,012 ID rows, the arm advances to
more seeds only if it meets all of these:
- macro pAUROC@20 gain ≥ 0.010;
- no source loses more than 0.010;
- macro Brier rises by no more than 0.005.

Nothing is promoted on one seed. OOD is not scored here. If the arm advances,
OOD becomes a separate request, given the earlier SHADE Arena regression.

**Stop** on input or runtime drift, nonfinite or missing gradients, failed
master, merge or serving parity, incomplete scoring, or completion. After
scoring, `run.py` releases the merged slot so the 50 GB container disk does not
fill.

```bash
PYTHONPATH=src:. <training-python> -m experiments.filtered_hard_labels.prepare --gpu-uuid UUID
PYTHONPATH=src:. <training-python> -m gleipnir.campaigns.monitoring prepare --config data/filtered-hard030-01/config.yaml
PYTHONPATH=src:. <training-python> -m experiments.filtered_hard_labels.run
python -m experiments.filtered_hard_labels.compare
```

## Status

Launched 2026-10-16 18:50 UTC on Pod `q7io6ui6gwbqo6` (`GPU-3d8d6bde…`). The
inventory passed, and the trainer resolved soft 0.7 / direct 0.3 at LR 5e-5.
The first updates have finite losses (0.67, 0.61).
