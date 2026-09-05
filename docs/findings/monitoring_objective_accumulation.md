# Objective accumulation audit

Date: 2026-09-05. Status: bug reproduced; corrected rationale reruns required.

The initial objective campaign's two rationale arms do not implement the frozen
relative loss weights. They must be retained as diagnostic artifacts and excluded
from selection. The three MIL-only arms do not have this particular bug.

The custom `AuxiliarySFTTrainer.compute_loss` returns microbatch means and ignores
`num_items_in_batch`. Transformers infers `model_accepts_loss_kwargs=True` from
the causal Qwen model's forward signature. When completion labels are present,
Trainer supplies a token count and assumes compute_loss has already normalized
over the accumulation window, skipping its usual accumulation division. The
sequential auxiliary backward explicitly divides by the window size, however.

For a window of K microbatches, the old gradient was therefore
`sum(direct gradients) + w/K * sum(rationale gradients)`, rather than
`1/K * sum(direct gradients) + w/K * sum(rationale gradients)`. At K=32 the
relative rationale weight was 32 times smaller than configured. The last window
contains 16 rows, so the factor there was 16. This is not just a logging issue;
it changes the objective balance and the gradient presented to clipping/AdamW.
The original one-microbatch preflight could not expose the problem.

`tests/test_auxiliary_accumulation.py` exercises the actual pinned Trainer and
Accelerate path on CPU with independent direct and auxiliary parameters. At
accumulation 32 and auxiliary weight 0.20, it reproduces gradients `(32, 0.20)`
before the fix and `(1, 0.20)` with explicit mean-loss normalization.

The fix explicitly sets `trainer.model_accepts_loss_kwargs=False` for this custom
mean-loss trainer and records `losses.accumulation_policy` as
`explicit_microbatch_mean_v1`. No-label direct/MIL batches already received the
usual division, so their normalization is unchanged. Historical completion-based
causal-model runs need separate audit; this finding alone does not invalidate
historical soft-only results or establish anything about their quality.

Corrected rationale runs retain the original data, seed, LR, one-epoch schedule,
adapter rank, memory recipe, and frozen ID promotion thresholds. They use new
artifact directories and are compared alongside the original valid MIL runs.
No ID result is used to choose which weights to rerun. Strict OOD is untouched.
