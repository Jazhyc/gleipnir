# Serving stack promotion check

Status: frozen 2026-11-27, before the B200 run.

**Question.** Does the optimized serving runtime, promoted from its
experiments into `gleipnir.serving.stack` (29 files moved, imports and
checkout lookups rewritten, no arithmetic change), serve the released
Gleipnir 4B and 27B exactly as before?

**Intervention.** This branch: recorded commands resolve onto the promoted
modules through `gleipnir.serving.sources.promote_command` and
`stack/promotion.json` (`status: candidate`). **Baseline:** `main` at
`1bff5636`, the code the accepted recipes were last run from. Both run on the
same Pod from `ghcr.io/jazhyc/gleipnir-serve:1.2` (its runtimes, kernels and
receipts), each from a clean code tree with the receipt-pinned
`src/gleipnir/__init__.py` (`b6acfaad…`) restored, as in the image.

**Measurements**, per model, in the image's default mode (larger 4B graphs,
Gigatoken), through the image entrypoint (`fetch`, `serve`):

1. `image/identity.py`: 248 synthetic prompts one at a time; the branch's
   scores are compared with `main`'s bit for bit.
2. `validate.py --splits id` (3,012 rows, campaign batching) against the
   recorded optimized rows; the release gates (MAE ≤ 0.005, correlation
   ≥ 0.999, |Δ source-macro pAUROC@20| ≤ 0.005).
3. `serving_concurrency/run.py --agent --gigatoken --levels 8 1`: the 96 agent
   trajectories, three passes (4B `fast_g4096e`, 27B `fast`).

**Acceptance rule (frozen).** Accept the promotion when, for both models,
(1) every identity score is identical to `main`'s, (2) the ID gates pass for
both phases, and (3) the branch's median one-request throughput is within
3% of `main`'s on the same Pod. Then set `promotion.json` to `accepted`
(user decision). **Stop conditions:** any startup failure other than the known
MXFP8 compile race, an identity difference, or a failed gate; report and stop
without retrying variants.

Outputs: `results/serving_stack_promotion/{main,branch}/`, throughput under
`results/serving_concurrency/promo_{main,branch}/`, logs in
`logs/runpod/serving_stack_promotion/`.
