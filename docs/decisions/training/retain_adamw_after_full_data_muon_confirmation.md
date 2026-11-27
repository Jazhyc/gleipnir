# Retain AdamW after full-data Muon confirmation

Date: 2026-08-13.
Status: historical; governs no future work.

- Neither predeclared Muon candidate passed the frozen three-seed confirmation
  gate. Muon `5e-5` reduced paired mean macro AUROC by 0.00103 and improved only
  one seed. Muon `1e-4` increased paired mean macro AUROC by 0.00097 with two
  seed wins, but regressed macro balanced accuracy by 0.00238 and macro Brier by
  0.00313, beyond the 0.002 tolerances.
- Retain rank-64 AdamW at `5e-5` as the selected optimizer recipe. Do not promote
  Muon based on its higher raw AUROC at `1e-4`, and do not tune additional
  optimizer rates on the confirmatory validation surface.
- Interpret the small effects cautiously. With three paired seeds, approximate
  95% t intervals for the AUROC differences include zero widely: [-0.00824,
  +0.00618] at `5e-5` and [-0.00597, +0.00791] at `1e-4`. The earlier 25%-data
  internal-screen advantage did not replicate robustly at full data.
