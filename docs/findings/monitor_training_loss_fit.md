# Matched 4B/9B online training-loss comparison

The saved regular 8,688-row runs show a modest training-fit advantage for 9B.
They provide evidence of a fitting difference under the existing recipe, but
cannot identify a hard 4B capacity ceiling or distinguish capacity from
optimization and pretrained representations.

Use the regular pair from
[student injection awareness](../../experiments/student_injection_awareness/README.md),
not the released mixed-data checkpoints or the strictly filtered 4B candidate.
Both regular models use identical trajectories, student instruction and Kimi K3
probabilities: one epoch, seed 0, rank 128/alpha 256, BF16 bases/FP32 adapters,
soft BCE only, AdamW 5e-5, linear decay, 3% warmup and logical batch 32.
Both complete 272 updates and all 8,688 rows. 9B uses checkpointing and a larger
adapter parameter count; backbone initialization and representational priors
necessarily differ. Historical metadata has null Git commit fields, so the
comparison binds saved receipt hashes rather than inferring missing source IDs.

| Statistic (nats) | 4B regular | 9B regular |
|---|---:|---:|
| Trainer mean online CE over the epoch | 0.233122 | 0.222823 |
| Approximate mean online KL | 0.076130 | 0.065831 |
| Mean CE over last 50 logged updates (221–270) | 0.204230 | 0.196459 |
| Last five-update logged CE (266–270) | 0.197231 | 0.188655 |

Mean teacher Bernoulli entropy is **0.156991554 nats**. Epoch KL subtracts
that entropy from the logged mean CE. It is approximate because Trainer
averages logical updates while entropy averages rows, and update 272 contains
only 16 examples. Do not subtract whole-population entropy from late-window
losses: their teacher uncertainty can differ. The 9B epoch CE is lower by
0.010299 nats; the last-50 gap is 0.007770. Approximate epoch KL is 13.5% lower
for 9B, so the advantage is measurable even though absolute CE differences are
small. This does not locate the mismatch in OOD-relevant examples.

These are online losses on changing checkpoints, before their corresponding
updates. Logged windows each cover five updates; the final two updates have no
separate loss report. They are not a fixed final-checkpoint pass over training,
and do not establish convergence. A larger model can learn faster or start
with better representations while the smaller model still has sufficient
capacity given a different optimization budget. Final-checkpoint training KL
and held-out teacher fidelity would be needed to distinguish those possibilities.

The [figure](../../figures/monitor_training_losses.png) shows raw five-update
reports and trailing 50-update averages, with whole-population teacher entropy
as a reference, not a lower bound for each individual batch. Regenerate with
`PYTHONPATH=src:. .venv/bin/python scripts/plot_monitor_training_losses.py`.
The script checks teacher hash/count, shared optimizer/loss settings, finite
logs, normalization, updates and full row counts before plotting. The adjacent
provenance file binds metadata, teacher targets, script and exports; the full
numerical report is also saved under `results/training_loss_capacity_audit/`.
No inference, retraining, model selection or compute lifecycle action occurred.
