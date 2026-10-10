# Model-specific injection projection after regular-data CAFT

User-requested follow-up: fit an injection vector in the completed regular-data
CAFT checkpoint and project it out at inference. Hypothesis: injection alarms
after training-only ablation remain accessible through a model-specific residual
direction. This is a post-training intervention, not another CAFT training run.

Use the existing `caft-regular-sdpa02` FP32 master and remote BF16 merge. Retire
its compiled scorer sequentially, preserve all results/weights/caches, and start
the existing BF16 eager Lens integration with adapter-specific master/native/
functional/operator gates. This is the established mechanistic-analysis serving
exception. No new capacity, merge, training or dependency changes.

Reuse exactly the original **64 fitting tasks / 896 rows / 384 paired
injected-minus-neutral contrasts**, with both honest and backdoored labels.
Capture all 32 layers; fit the unit mean difference at layer 20. Apply that
single direction after all 32 layers and at all tokens, beta 1, zero centers,
using the unchanged inference projection implementation. Freeze this geometry
before outcomes: no layer/beta/template/seed sweep, extra intervention arms,
checkpoint selection or promotion. The shared fitter serializes unused controls;
only full projection is evaluated.

Score fresh plain and projected arms in the same eager engine on the original
**591 disjoint test tasks / 8,218 rows**, excluding every fitting variant.
Keep c64, batches 128 and the same 128-task full-readout diagnostic subset.
Report each arm's clean-only at-most-2%-FPR threshold and the fixed plain cutoff,
injected-honest FPR, backdoor recall, matched AUROC/raw pAUROC20, vanilla/neutral
controls, calibration/ties, answer mass and paired score shifts. Compare the
new vector's cosine/norm with the original augmented and filtered-model axes.
Compare cached same-partition projection results on those models, with their
different data/training interventions disclosed. Compare the fresh plain arm
with CAFT's compiled BF16 scores as a serving-mode diagnostic.

Stop on identity/runtime/source/input drift, failed master/native/operator gates,
fit/test leakage, nonfinite captures/scores, missing coverage, cleanup failure or
OOM. Preserve partial/failed receipts and exact completed batches. Keep a passing
Lens engine warm and all weights remote; no capacity lifecycle. Monitoring is
active-turn only. This is one seed on previously investigated APPS templates;
there is no requested ID evaluation or claim of untouched confirmation.

`prepare.py` freezes `startup.json` and `config.json`; `run.py` reuses the
existing Lens startup, capture/fitter/scorer and projection geometry validator.
Artifacts use `results/caft_direction_refit/caft_refit01/` and startup receipts
`results/b200_sdpa_lens/caft_refit01/`; logs use `logs/runpod/caft_direction_refit/`.
