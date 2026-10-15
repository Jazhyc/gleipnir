# Precision and kernel choices may be selected on the OOD suite

User decision, 2026-10-15. Training- and serving-precision or kernel choices
may use the frozen six-source OOD suite for selection. This covers MLP
precision, attention implementation, quantized-serving recipes and QAD teacher
configuration. Objective, data, filtering, prompt and threshold ablations stay
ID-only under the [CoT-removed ID contract](cot_removed_id_validation.md).

Rationale. The OOD suite has already been consulted repeatedly. Precision
effects concentrate on low-margin examples, which pAUROC@20 weights heavily.
OOD inputs have smaller margins and less familiar activation statistics, so ID
underestimates these effects: the same QAD checkpoint loses 3.8 ID points but
4.7 OOD points under optimized serving, and its 71% ID recovery does not carry
to OOD. Training precision also changes the learned weights, so this scope is a
chosen convention rather than a mechanistic guarantee.

Conditions:
- Freeze the selection rule before scoring candidates.
- Report every candidate, including per-source results.
- Measure seed variation before treating differences between candidates as
  real.
- Disclose that OOD values for selected precision configurations are
  optimistically biased.

The frozen inputs remain unchanged under
[the comparability decision](frozen_ood_comparability.md).
