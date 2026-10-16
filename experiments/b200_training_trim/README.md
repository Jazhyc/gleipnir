# Final-layer trim in training

The monitoring loss reads one hidden state per packed example: the last token
after layer 31, which is a full-attention layer. Training currently runs that
layer on every token. For non-decision tokens it only needs to produce keys and
values. The serving equivalent left scores bitwise unchanged
([finding](../../docs/findings/b200_last_layer_trim.md)).

`gleipnir.training.last_layer_trim` changes what layer 31 computes on the
packed path:
- **Every token** still gets the input norm and the K/V projections.
- **Decision positions only** get the gated q projection, attention (one
  masked SDPA call: each decision query sees only its own sequence), the o
  projection, the MLP and the norms.
- **Other output rows are zero.** The LM head reads only decision positions
  (`logits_to_keep`).

Gradients are mathematically unchanged; only the attention kernel call differs.
On a tiny fp32 Qwen3.5, a CPU test matches logits, loss and every gradient to
within 1e-5.

**Expected gain.** The trim removes 102M of the 3.57G linear MACs per token
(2.9%), plus one of eight full-attention layers' quadratic attention. FlashQLA
and other memory-bound kernels do not shrink, so a 2–3% shorter update is the
plausible range.

## Design

**Arm.** `training-trim-s1-01` copies `teacher-seed-sdpa-s1-01` (seed 1,
strict-filtered BF16/SDPA, 245 updates) and adds `last_layer_trim: true`.

**Canary.** Before any update, the trainer compares trim off and on, with
gradients cleared and no optimizer step. It runs two cases: one real
multi-sequence pack filling the 16,384-token budget, and the longest single
example. It also measures a trim-off repeat as the noise floor. It requires:
- finite logits and gradients;
- a nonzero gradient on every layer-31 LoRA parameter;
- global gradient relative L2 ≤ 1%;
- decision-logit maximum absolute difference ≤ 0.125, which is two BF16 ulps
  at |logit| 8–16.

Otherwise training stops before the first update. Completed-run validation
requires a passed canary and nonzero trimmed calls.

**Timing control.** The T=2, T=4 and hard 0.3 runs used the same GPU
(`GPU-3d8d6bde…`) and the same seed-1 partitions. Their arithmetic differs only
in the loss scalar. `compare.py` first requires every physical microbatch to
match, then compares updates 11–245 against the median of those three runs.

**Adoption.** This is a systems check with no quality selection. Adoption
requires all of these, and is the user's decision:
- the canary passes;
- the steady median update is at least 2% shorter;
- ID scores agree with the seed-1 control at the level of training arithmetic
  noise. Score MAE and pAUROC are reported; there is no promotion.

**Stop** on a failed canary, drift, nonfinite or missing gradients, failed
parity, incomplete scoring, or completion. After scoring, the run releases its
merged slot.

```bash
PYTHONPATH=src:. <training-python> -m experiments.b200_training_trim.prepare --gpu-uuid UUID
PYTHONPATH=src:. <training-python> -m gleipnir.campaigns.monitoring prepare --config data/training-trim-s1-01/config.yaml
PYTHONPATH=src:. <training-python> -m experiments.b200_training_trim.run
python -m experiments.b200_training_trim.compare
```
