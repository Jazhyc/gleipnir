# One strict-filtered partial-AUROC variant

Hypothesis: soft Kimi distillation plus a small source-macro negative-tail
ranking term improves ID raw normalized pAUROC@20 and transfers to OOD.
One predeclared variant only: loss = soft BCE + 0.1 * pairwise softplus,
temperature 1, beta 0.2. Within each full logical batch of 32 and each source,
compare every harmful example with the highest-scoring 20% of benign examples.
Use fractional tail mass and share boundary ties equally. Average sources that
contain both classes; single-class groups retain ordinary soft BCE only.
This is a minibatch partial-AUC surrogate, not an exact global pAUROC objective.
Small per-source tails have sampling variance; no tail/weight/seed sweep.

Preserve the strict delta-zero training population (7,836 rows, 75,851,675
gradient-pass tokens), original source labels, teacher probabilities, seed 0,
zero-B FP32 rank-128 initialization, BF16/SDPA/FlashQLA, learning rate 5e-5,
one epoch, logical batch 32, 245 updates and original packing budget/context cap.
No augmentation, model continuation or extra data. Reuse the completed strict
checkpoint as the control; do not train another baseline.

Long trajectories require physical microbatching. First collect all logical-batch
scores without gradients using the same partitions and model state. Compute
the logical ranking loss and its derivatives, then replay the ordinary forward/
backward with those exact derivatives. No optimizer update occurs between passes.
Dropout is zero. Check every replayed margin against its score-pass value and
fail above 1e-5. Record both passes, extra forward tokens, rank loss, active
groups and finite gradients. Toy full-versus-replayed parameter gradients must
match before training. The existing model/kernel startup receipt is reused;
the changed objective receives these fresh targeted checks. Score-pass overhead
means matched data/updates do not imply matched allocated GPU time.

Select only the final epoch; no early stopping, adaptive lambda or OOD tuning.
Evaluate all canonical 3,012 ID and 6,395 OOD examples, including negative results.
Use merged BF16 backbone with the validated FP32 final two-row head and compare
the archived strict model's matched `fp32-monitor-head01` outputs. Report source/
pooled ranking, calibration, thresholds, ties and throughput. One seed does not
establish robust transfer. Historical Aletheia pairwise overfitting remains a
qualification, not a reason to tune this variant after seeing results.

The detached driver trains, validates, merges into the existing replaceable
ephemeral slot, checks adapter-specific master/merge/serving parity and evaluates.
Keep original masters/receipts and persistent caches; stop on drift, replay or
finite-gradient failures, missing scores, failed parity or the three-hour stage
deadline. No automatic retraining or failed-parity waiver. Retire the owned
scorer at completion/failure, but keep the Pod running. The prior unattended
preference applies; no agent scheduler is available.

The `strict-pauc01` startup stopped at offline tokenizer loading because its
launcher omitted the persistent Hugging Face cache paths. No model loaded or
optimizer update ran. `strict-pauc02` preserves that failed attempt and uses
the identical objective/data/recipe with corrected cache/PATH bindings.

```bash
PYTHONPATH=src:. python -m experiments.partial_auroc.prepare
PYTHONPATH=src:. python -m experiments.partial_auroc.run
```
