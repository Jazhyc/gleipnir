# Plot Gleipnir at optimized-serving performance; no QAD for the 27B

Date: 2026-10-20. Status: recorded in the decision log.

- User decision: the frontier registry enters Gleipnir models at their
  performance on the project's optimized serving stack, because project
  serving prices will set their cost later. Until then the dated proxy cost
  is kept.
- Gleipnir 27B is entered at `0.9508` Mean-OOD pAUROC@20 (full teacher
  prompt, `QuantFullWorker`, no QAD; BF16 `0.9352`). It now dominates Gemini
  3.1 Pro and Claude Opus 4.6, and the frontier is pplx-decider v1.2 and
  Gleipnir 27B. The gain is almost entirely SHADE-Arena, which is kept to
  match the Sinha et al. benchmark format.
- Gleipnir 27B needs no QAD: optimized serving costs it nothing measurable
  (`−0.0025` under the student prompt, `+0.0156` under the teacher prompt).
  QAD effort goes to Gleipnir 4B, which loses `0.0529` under optimized
  serving ([optimized gap](../../../experiments/serving/optimized_gap/README.md)).
- Update, same day: Gleipnir 4B v0.3 (v0.2 plus QAD) is entered at its
  optimized-serving score, `0.7703` (BF16 `0.8039`). QAD recovered 98% of the
  ID loss but 23% of the OOD loss
  ([QAD](../../../experiments/quantization/qad_v02/README.md)).
