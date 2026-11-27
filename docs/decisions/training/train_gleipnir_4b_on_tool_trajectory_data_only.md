# Train Gleipnir 4B on tool-trajectory data only

Date: 2026-10-20. Status: recorded in the decision log.

- User decision: Gleipnir 4B uses the same training data as Gleipnir 27B. That
  is the strict-filtered tool-trajectory monitoring pool
  (`data/teacher-disagreement-delta000-sdpa04`, 7,836 rows, Kimi K3 soft
  targets), with none of the 13,149 deception rows (Aletheia's Quest, Liars
  Bench). One recipe for both sizes is simpler to explain.
- Rationale: the deception data's measured gains are within seed noise. At 9B
  it gave +1.15 ID points and −0.15 OOD; at 4B, +0.61 OOD against a 4B OOD seed
  SD of 1.8 points. Both are single seeds.
- The released mixed-data Gleipnir 4B (`0.7823` Mean-OOD) stays a historical
  measured point. Replacing the release is a separate step, selected on ID only.
