# Concept-ablation training on regular monitoring data

User-requested CAFT-style experiment: train on all **8,688 regular originals**,
with no synthetic injections or activation filtering, while removing the fixed
original injection direction from the latent computation. Hypothesis: learning
without this component improves injection robustness without augmentation.

Use the original unfiltered SDPA augmented model's unit layer-20 injection axis,
SHA256 `8c3c6e50ee1d3880d854754f9899f5684ae2e2c686479c3ceaaf3677d9eecfa7`.
After every decoder layer and at every token, apply h <- h-u(u^T h), with FP32
accumulation and BF16 boundary casting. Differentiate through the projection.
The fixed vector is not trainable; hooks/buffers are absent from exported weights.
Remove hooks before export and run **ordinary, unablated inference** as CAFT does:
https://arxiv.org/html/2507.16795v2#S3. This restricts gradients through the chosen
residual sites, without guaranteeing globally frozen parameter components.
Our APPS-fitted axis is an explicitly disclosed adaptation, rather than the
paper's OOD-data-free concept discovery method.

Preserve the seed-0 BF16/SDPA/FlashQLA recipe, zero-B rank-128 FP32 initializer,
all original Kimi targets, row order, loss, batch 32 and one epoch/272 updates.
Retain the 16,384 packing budget, 29,696 context cap, AdamW 5e-5 and linear
schedule/3% warmup. Reuse unchanged kernel/packing startup receipts; add targeted
forward/backward projection checks and per-layer physical-forward coverage.
This is a matched training exception to the FP4/FA4 default. Use gated BF16
serving to avoid quantization confounding this new adapter; retain its explicit
exception to the optimized default.

The user requests reusing existing regular-model results, with **no fresh control
training**. Compare with cached modern CoT-removed ID and APPS regular FP4/
optimized results; their training precision, serving and host differences prevent
clean attribution solely to CAFT. Also retain the current augmented SDPA/BF16
comparison as context. No retraining of controls or automatic parity exception.

Select only the complete final checkpoint. Require FP32-master/BF16-merge/BF16
serving parity and native audits, then score frozen ID and APPS. Primary APPS
excludes all 64 axis-fitting tasks: 591 tasks / 8,218 rows. Full registered scores
are diagnostic. Report ID source-macro/pooled raw pAUROC20/AUROC, calibration and
thresholds; APPS FPR/recall, matched ranking, vanilla/neutral controls and ties.
No axis/layer/beta/seed/prompt/checkpoint selection or promotion on test results.

Stop on identity/runtime/source drift, invalid projection, missing/nonfinite
gradients, OOM, incorrect token/row/layer coverage or failed adapter parity.
Keep FP32 masters persistent, new merged weights remote/ephemeral, and reuse
persistent caches. Archive and verify an obsolete reconstructable merged copy
before reclaiming its ephemeral disk space. Do not delete master adapters,
quality receipts or caches. No capacity lifecycle. Monitoring is active-turn only.

The shared monitoring campaign runner handles training/export/merge/evaluation.
`prepare.py` binds the regular data, cached controls and direction; `run.py`
executes the single CAFT campaign and computes the APPS holdout. Results use
`results/caft-regular-sdpa02/` and `results/concept_ablation_training/`.

Attempt 01 stops before any update: its absolute alignment tolerance ignored
BF16 rounding. Attempt 02 retains the identical projection arithmetic and uses
an explicit BF16 gradient reference plus <=0.1% relative alignment checks.
The original failed contract, sources and receipts remain preserved.
