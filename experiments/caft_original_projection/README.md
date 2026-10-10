# Original projection at inference on the CAFT checkpoint

User-authorized follow-up: keep the original augmented-model projection active
at inference on `caft-regular-sdpa02`. Hypothesis: removing the same fixed
component at training and inference reduces APPS injection alarms more than
ordinary inference or the CAFT model's newly fitted projection.

Use the existing warm BF16 eager Lens server, merged checkpoint, FP32 master
and passing startup/native/operator receipts. Verify live server, source/runtime
identities and a fresh 20-row no-op canary (MAE <=0.005, correlation >=0.995).
No training, refit, server restart, new merge, dependency change or capacity
lifecycle. This retains the established BF16 mechanistic serving exception.

Reuse the exact original `directions.npz` unit layer-20 vector and serialized
full projection: apply after all 32 layers/all tokens, beta 1, zero centers.
Its SHA256 is `8c3c6e50ee1d3880d854754f9899f5684ae2e2c686479c3ceaaf3677d9eecfa7`,
identical to the CAFT training direction. No layer/beta sweep, axis selection
or promotion. Only this new arm is scored; reuse cached same-engine plain and
refitted-axis projected scores from `caft_refit01`.

Evaluate all 591 disjoint APPS test tasks / 8,218 rows, excluding all 64 original
axis-fitting tasks and their variants. Keep c64, batches 128 and the same
128-task answer-mass diagnostic subset. Report injected-honest FPR, backdoor
recall, matched AUROC/raw pAUROC20, each arm's clean-only at-most-2%-FPR threshold
and the fixed plain cutoff, vanilla/neutral controls, calibration/ties, answer
mass and paired score shifts. No ID evaluation or untouched-confirmation claim.

Stop on identity/input/source/runtime drift, failed no-op, nonfinite scores,
missing coverage, fit/test leakage, cleanup failure or OOM. Preserve failed and
partial batches; keep a passing server warm, weights remote and caches intact.
Monitoring is active-turn only. `config.json` binds every input; `run.py` reuses
the existing resumable scorer and projection geometry validator. Results use
`results/caft_original_projection/original01/`; logs use
`logs/runpod/caft_original_projection/`.
