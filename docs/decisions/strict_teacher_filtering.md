# Strict teacher agreement is the default training-data filter

User selection, 2026-10-11: default future binary tool-trajectory distillation
to strict teacher/source-label agreement filtering. Remove a training row when
`int(p_harmful >= 0.5) != source_label`; a score exactly 0.5 predicts harmful.
Join cached targets by dataset/index and validate teacher prompt provenance.
Preserve retained row bytes/order, source labels and soft probabilities.
Apply only to training; evaluation populations remain frozen.

The shared monitoring campaign's `prepare` and `run` paths materialize this
selection by default. They preserve the original config and inputs, write a
resolved config and paired subsets under `data/<campaign>/strict_filter/`,
recount untruncated decision-boundary tokens, and freeze the retained population
and one-epoch update count. Record removed IDs/counts, source/class composition,
input/output checksums and original augmentation audit. Augmented inputs are
audited before filtering. Resume binds the original and resolved artifacts.
Preparation needs local base tokenizer assets, but does not load model weights.

Set `teacher_filter: none` explicitly for an unfiltered control or another
predeclared filtering study. Unknown policy names fail. Historical prepared
campaigns keep their original contracts and source snapshots. Legacy standalone
experiment launchers are frozen reproduction paths; new campaigns should use
the shared runner or explicitly apply this selection during preparation.
Filtering derived views must preserve their parent lineage for grouped holdouts;
the rule does not silently remove entire groups beyond the specified rows.

The [ID sweep](../findings/teacher_disagreement_filtering.md) removes 852/8,688
rows (9.81%), reducing training from 83.82M to 75.85M tokens and 272 to 245
updates. Strict filtering improves one-seed ID ranking but increases false
alarms at threshold 0.5. The [OOD follow-up](../findings/teacher_disagreement_ood.md)
finds a small, uneven ranking gain against historical cached references, with
mixed-data and score-resolution qualifications. This default is the user's
selection for data efficiency and observed ranking; it does not establish
universal quality improvement or calibration equivalence. Keep negative results
and evaluate source-specific calibration and transfer for new campaigns.
