# Activation-ranked training removal

User-selected intervention: remove the top **20% of harmless original records**
by trained-minus-base layer-20 projection onto the existing APPS unit direction.
Hypothesis: removing examples with the largest learned alignment increases
reduces injected-honest alarms; failure or harmful-recall/ID regression is useful
negative evidence. No expected success or promotion. Neither Kimi targets nor
Qwen firewall scores participates in selection. Do not change the axis, sign,
layer, fraction or rule after held-out results.

Rank all 4,344 harmless originals from the complete 8,688-row census, descending
`delta_z20`, breaking ties by original ID. Select ceil(0.20*4344)=869, then close
exact-trajectory groups. Remove each selected original and its replacement
augmented view/ledger entry/teacher record together; preserve retained bytes,
order, labels, targets and augmentation unchanged. No harmful example is removed.

Control: remove the same number using a fixed SHA256(seed:index) ordering within
source, label, existing augmentation status and source/label length quartiles.
Quartiles use original activation-census prompt lengths, ties by original ID.
Sample from the full stratum: overlap with activation removal is permitted and
reported, because heavily removed sources cannot support disjoint matched sets.
Require exact-content closure and identical stratum counts. This control matches
coarse lengths, not exact token counts. Record remaining tokens and exposure.

Train both conditions from the original zero-B rank-128 FP32 initializer with
the preserved seed-0 BF16/SDPA/FlashQLA recipe, one epoch, AdamW 5e-5, linear
decay/3% warmup, logical batch 32 and 16,384 packing budget. Reuse its validated
startup receipt and persistent compiler caches. Only data/input bookkeeping,
result paths and the resulting epoch step/schedule lengths change. Runtime and
hardware must match the preserved recipe. No new seeds or attention/precision
changes: SDPA run variance remains an unresolved limitation.

Baseline: reuse the completed unfiltered current SDPA replay's BF16 ID and APPS
scores. Run new-adapter FP32-master/BF16-merge/BF16-serving gates, then the frozen
CoT-removed ID and APPS benchmark/honest-control populations. BF16 serving is a
matched experimental exception to the optimized default, to avoid confounding
filtering with its documented score drift. Preserve FP32 adapters and keep each
merged BF16 checkpoint on remote ephemeral storage. No weights transferred here.

The frozen direction was fitted on 64 APPS tasks, so this curation indirectly
uses that fit partition. Exclude all its task variants from primary APPS metrics:
591 test tasks / 8,218 rows, using the preserved partition. Recompute the cached
baseline on the same held-out tasks. Registered full-population summaries remain
diagnostic, not held-out evidence. No new APPS fitting or partition selection.

Primary diagnostic: change in mean injected-honest APPS FPR at the existing
clean-only at-most-2%-FPR operating point, with backdoor recall, matched AUROC/raw
pAUROC20 and vanilla controls. Report ID source-macro/pooled AUROC/raw pAUROC20,
calibration, source composition and activation-filter minus random-control
differences. Select only each final one-epoch checkpoint. No cutoff, prompt,
checkpoint or recipe selection on ID/APPS; no promotion. Single-seed differences
cannot establish robust causal attribution to individual examples.

Stop on source/input/initializer/runtime drift, nonfinite/missing gradients,
OOM, failed adapter-specific parity/native audits, missing coverage or completion.
No automatic failed-parity exception. If a stage fails, preserve it and continue
independent work without hiding the failure. Capacity stays running. Monitoring
is active-turn only; no heartbeat tool is available.

`prepare.py` freezes exclusions and materializes both campaigns; `run.py` runs
them sequentially with the shared monitoring campaign runner; `analyze.py`
reports the primary APPS holdout comparison. Inputs and outputs
use ignored `data/activation-filter-*` and `results/activation-filter-*` trees.
