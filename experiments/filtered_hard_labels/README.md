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

## Result (2026-10-16, EU-RO-1 `GPU-3d8d6bde…`)

The run lasted 18:10–19:15 UTC. Training, master, merge and serving parity all
passed, and the merged slot was released after scoring.

| ID, 3,012 rows | pAUROC@20 | Gloom | STRIDE | AUROC | Brier |
|---|---:|---:|---:|---:|---:|
| control (seed 1) | 0.8978 | 0.8168 | 0.9788 | 0.9627 | 0.0722 |
| seeds 0 / 2 (context) | 0.8959 / 0.9142 | 0.829 / 0.862 | 0.963 / 0.967 | 0.964 / 0.973 | 0.089 / 0.061 |
| **hard 0.3** | **0.9056** | 0.8383 | 0.9729 | 0.9701 | 0.0633 |
| Δ vs. control | **+0.78** | +2.15 | −0.59 | +0.0074 | −0.0090 |

**The arm does not advance under the frozen rule.** Its gain of 0.78 points is
below the 1.0-point bar. No source loses more than 1.0, and Brier improves.

- **Within seed range.** The result sits inside the seed 0–2 range
  (0.896–0.914). On one seed it is consistent with a small benefit or none.
- **The prediction was wrong on Gloom.** I expected the 0.3 mix to act like a
  milder version of sharpening, which lost 13.6 Gloom points. It gained 2.1
  instead. So the mix and sharpening do not act alike on Gloom, even though
  both mostly move the less confident targets.
- **Smaller effect than the old screen.** The old unfiltered 2e-5 screen gained
  2.8 points. That fits the idea that most of that gain came from overriding
  conflicting targets, which the strict filter now removes. But this is one
  seed, and the LR differs.
- **Diagnostics.** Mean absolute score difference vs. control is 0.082, with 239
  threshold flips. Median |margin| is 2.88, and 16% of rows have |margin| > 4.

Receipts: `results/filtered_hard_labels/{comparison,status}.json` and
`results/filtered-hard030-01/`.
