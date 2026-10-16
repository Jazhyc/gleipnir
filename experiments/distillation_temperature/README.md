# Distillation temperature

Every Kimi K3 student so far trains with temperature 1: binary BCE between the
student's `1`−`0` logit margin and the teacher probability. This experiment
asks whether another temperature improves held-out ID ranking.

## Prior evidence

- **Old H100 recipe** (internal dev split, macro AUROC, seed 0,
  [optimization screen](../optimization_regularization_screen/README.md)).
  Teacher-only flattening at logit scales 1.5 and 2 lost 0.0009 and 0.0020
  AUROC.
- **Margin Huber loss** (δ 0.5 / 1.0, close to the high-T limit;
  [structural screen](../training_procedure_screen/README.md)). It moved AUROC
  by −0.0010 and +0.0001, inside the 0.002–0.004 seed spread.
- **Hard-label mixing** ([finding](../../docs/findings/monitoring_hard_labels.md)).
  A 0.3 hard fraction gained 2.8 ID pAUROC@20 points, but only 0.55 OOD points,
  with a SHADE Arena regression. That run used unfiltered targets.
- **Never tested:** sharpening (T < 1) and classic distillation temperature
  (both sides divided by T, loss × T²).

## Target audit (`audit.py`, `target_audit01`)

The audit covers the 7,836 strict-filtered training targets
(`teacher.jsonl`, `ba8b8b9c…`).

**Teacher targets**
- **Bimodal in probability.** 26% of targets fall below 0.01 and 29% above
  0.99. 76% lie outside 0.05–0.95 and only 7.5% within 0.2–0.8.
- **Not saturated in logit space.** No target is exactly 0 or 1. Margins span
  −10.5 to +11.8 with median |margin| 4.9, and 65% have |margin| > 4.
- **Coarsely quantized.** 99% of margins lie on a 1/8 grid, and there are only
  198 distinct values. Almost every row shares its exact target with other
  rows, so the teacher cannot order rows within a tie.
- **Agrees with the labels.** Teacher side and source label disagree on only
  28 rows.
- **By source.** Gloom (median |margin| 3.9) and BashArena (4.1) are the least
  confident. BashBench (6.0) is the most.

**Gradient share at T=1.** For a small student error, the T²-scaled gradient is
proportional to σ′(z_t/T). Rows with |z_t| > 4 carry this share of it:

| T | 0.5 | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|---:|
| share of gradient on the 65% confident rows | 0.2% | 8.7% | 36% | 55% | 62% |

**Student margins.** The seed 0–2 students' ID margins are compressed relative
to the teacher's training margins:
- median |margin| is 2.25–2.6, against the teacher's 4.9;
- only 11–16% of rows have |margin| > 4, against the teacher's 65%;
- the extremes are about ±7, against the teacher's ±12.

The two populations differ (ID vs. training), so this comparison is
descriptive. Even so, it fits a T=1 objective that barely supervises the
ordering among confident rows, which is the region pAUROC@20 reads.

## Design (frozen before training)

**Hypothesis.** Changing the soft-loss temperature improves held-out ID macro
pAUROC@20 over the seed-matched T=1 control.

**Arms.** Each arm copies `teacher-seed-sdpa-s1-01` (seed 1, strict-filtered
BF16/SDPA, 245 updates, LR 5e-5). It changes only its knob, plus identity, GPU,
merge slot and serving hash.

| Arm | Knob | Effect |
|---|---|---|
| `kd-temperature-t2-01` | `soft_kd_temperature: 2` | same optimum; 36% of the gradient on confident rows |
| `kd-temperature-t4-01` | `soft_kd_temperature: 4` | same optimum, near logit matching (55%) |
| `kd-sharpen-t050-01` | `soft_target_logit_scale: 0.5` | targets σ(2·z_t), closer to hard labels |

**Control.** The control is the existing seed-1 run on the same EU-RO-1 machine
(`2mj1yzztlz17`). It used a different GPU on that machine (`e5eb9962…` vs.
`3d8d6bde…`). Seeds 0 and 2 give context: ID SD is 1.0 point.

**Selection.** ID only (3,012 rows, BF16 vLLM 0.31). An arm advances to seeds
0 and 2 only if it meets all of these, measured against the control:
- macro pAUROC@20 gain ≥ 0.010;
- no source loses more than 0.010;
- macro Brier rises by no more than 0.005.

Nothing is promoted on one seed, and OOD is not scored.

**Stop** on input or runtime drift, nonfinite or missing gradients, failed
master, merge or serving parity, incomplete scoring, or completion.

## Implementation

`soft_binary_distillation_losses` takes a `kd_temperature` argument, which
`student.training.soft_kd_temperature` sets. With T = 1 the loss is bitwise the
original BCE. Campaign configs pass `soft_kd_temperature` and
`soft_target_logit_scale` through to the trainer. Training metadata records
both, and completed-run validation checks them.

```bash
PYTHONPATH=src:. <training-python> -m experiments.distillation_temperature.prepare --gpu-uuid UUID
for a in kd-temperature-t2-01 kd-temperature-t4-01 kd-sharpen-t050-01; do
  PYTHONPATH=src:. <training-python> -m gleipnir.campaigns.monitoring prepare --config data/$a/config.yaml
done
PYTHONPATH=src:. <training-python> -m experiments.distillation_temperature.run
```

## Results (2026-10-16, EU-RO-1 `GPU-3d8d6bde…`)

None of the three arms advances. T = 1 stays the soft-loss temperature. Every
arm scores below all three baseline seeds (0.896–0.914).

| ID, 3,012 rows | pAUROC@20 | Δ vs. control | Gloom Δ | STRIDE Δ | Brier Δ | median \|margin\| |
|---|---:|---:|---:|---:|---:|---:|
| control (seed 1) | 0.8978 | — | (0.8168) | (0.9788) | (0.0722) | 2.25 |
| KD T=2 | 0.8789 | −1.89 | −0.94 | −2.85 | +0.002 | 2.88 |
| KD T=4 | 0.8802 | −1.76 | +0.01 | −3.53 | +0.004 | 3.25 |
| sharpen ×2 | 0.8291 | **−6.87** | **−13.64** | −0.10 | +0.019 | 3.38 |

Δ columns are in pAUROC@20 points, except Brier. The control row shows absolute
values in parentheses.

- **Distillation temperature.** Higher T did what the audit predicted for the
  margins. The share of ID rows with |margin| > 4 rose from 12% to 24% (T=2) and
  31% (T=4), and unique scores rose from 98 to 117 and 140. Ranking still got
  worse, mainly on STRIDE.
  - So compressed margins are not what limits the student. Weighting the
    confident rows more costs ranking quality on this ID set.
  - The drop is about two seed SDs on one seed, but both arms agree.
- **Sharpening.** It loses 13.6 Gloom points, far beyond seed noise, while
  STRIDE is unchanged.
  - Gloom's teacher targets are the least confident (median |margin| 3.9).
    Doubling them discards the graded uncertainty that the Gloom ranking
    depends on.
  - This differs from the earlier 0.3 hard-label mix, which was unfiltered, used
    LR 2e-5 and gained 3.2 Gloom points. Sharpening every target is not a
    substitute for that mix.
- **Scores vs. control.** Mean absolute score difference is 0.070, 0.090 and
  0.110, with 183, 229 and 294 threshold flips.

**Operational note.** The sharpening arm trained fully, but its merge failed
with `No space left on device`. The two finished arms' merged checkpoints were
still occupying the 50 GB container disk. `recover.py` handled it:
1. It retired those checkpoints after the `release_merge` checks.
2. It archived the failed receipt and log (`runner_failed_merge_attempt.json`,
   `merge_failed_attempt.log`).
3. It resumed the campaign from the merge step. Training was not repeated.

Later multi-arm queues on this Pod should release each merged slot after
scoring.

Receipts: `results/distillation_temperature/{comparison,status}.json` and
`results/<arm>/`.
