# Final-layer trim in training

The monitoring loss reads one hidden state per packed example: the last token
after layer 31, which is a full-attention layer. Training currently runs that
layer on every token. For non-decision tokens it only needs to produce keys and
values. The serving equivalent left scores bitwise unchanged
([finding](../../docs/findings/b200_last_layer_trim.md)).

`gleipnir.training.last_layer_trim` changes what layer 31 computes on the
packed path:
- **Every token** still gets the input norm and the K/V projections.
- **Decision positions only** get the gated q projection, attention (the same
  Transformers SDPA entry point the untrimmed router uses, with one query per
  segment), the o projection, the MLP and the norms.
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
- a gradient on every layer-31 LoRA parameter that the untrimmed reference
  trains (zero-initialized LoRA B leaves A without gradient in both modes);
- global gradient relative L2 ≤ the trim-off repeat's noise floor + 1 point;
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

## Run log

**Full campaign (stopped).** The run started 19:39 UTC and was stopped during
model load, at the user's request, in favour of the ten-update benchmark. No
canary or update ran.

**Benchmark control.** The first launch used the baseline config's
`packing_only: false`, which also times padded batching. It was stopped after
12 minutes and is kept as `screen_control_padded_stopped`. The packed-only
control completed 19:58–20:08 UTC with the same partitions as the recorded
baseline: **4.924 s mean update** (median 4.827 s), against 5.145 s on NC2. This
host is about 4% faster, so the trim is compared with this control, not the
recorded number.

**Trim canary, first attempt.** The pack builder took examples in dataset
order and stopped short of two sequences (`screen_trim_failed01`). It now packs
the shortest examples first.

**Trim canary, second attempt (failed).** This version used one masked SDPA
call.

| case | trim vs. reference, grad rel. L2 | trim-off repeat (noise) | logit max diff |
|---|---:|---:|---:|
| 16,099-token pack, 65 sequences | 1.01% | 0.43% | 0.125 |
| longest example, 28,733 tokens | 1.09% | 1.30% | 0 |

The trim did not lose any trained gradient. The flagged layer-31 LoRA A
matrices have zero gradient in both modes because their B matrices start at
zero. Two changes followed, both fixed before the next run:
- the attention now uses per-segment Transformers SDPA, matching the untrimmed
  kernels;
- the gates are relative to the reference and to its noise floor, because
  BF16 backward on this recipe is not run-to-run deterministic.

**Third attempt (`screen_trim_failed03`).** The trim canary passed:

| case | trim vs. reference | noise floor | logit max diff |
|---|---:|---:|---:|
| 16,099-token pack, 65 sequences | 1.10% | 0.43% | 0.125 |
| longest example, 28,733 tokens | 1.14% | 1.01% | 0 |

Per-segment SDPA did not shrink the packed-case excess. The screen's compiled
packing-isolation check then stopped on the trim's own guard: its reference
runs each example unpacked, which the trim rejected. Unpacked calls now fall
back to the exact untrimmed layer and are counted. The isolation check
therefore compares trimmed packs with untrimmed singletons, which is a second
check of the trim.

## Result (fourth attempt, 2026-10-16 20:19–20:24 UTC)

The run used the same GPU as the control, the same 10 measured batches and
identical physical partitions:

| packed benchmark | control | trim | change |
|---|---:|---:|---:|
| mean update | 4.924 s | 4.684 s | **−4.9%** |
| median update | 4.827 s | 4.585 s | −5.0% |
| peak allocated | 147.14 GiB | 143.38 GiB | −3.8 GiB |
| new Dynamo graphs during measured updates | 0 | 0 | |

- **Speed.** Every measured batch is faster, by 4.3–5.4%. That is above the
  FLOP estimate, because the trim also removes layer 31's full-size zero-grad
  backward and its stored activations.
- **Learning.**
  - The 10 losses track the control within 0.002–0.018: step 1 is
    0.6035 vs. 0.6005, and step 10 is 0.3325 vs. 0.3254.
  - Gradient norms track as well.
  - The after-training probe loss is 0.7780 vs. 0.7794.
- **Isolation.** The compiled packing-isolation check passed, with repeat,
  perturbation and cross-example gradients all 0. Packed (trimmed) vs. singleton
  (untrimmed) gives a gradient relative L2 of 0.0556 and loss 0.9398 vs. 0.9576;
  the control gives 0.0061 and equal losses. That exceeds the strict 0.05 gate
  but is accepted under the configured 0.15 learning tolerance. The likely cause
  is that the trimmed layer runs eager while the singleton reference runs the
  compiled layer: eager vs. compiled moves the probe loss by 0.01–0.02 in both
  runs. The trainer canary, which compares trimmed and untrimmed packs under
  the same compile state, shows 1.10% against a 0.43% noise floor.
- **Trimmed calls.** The run made 195 trimmed calls and 2 unpacked untrimmed
  calls.
