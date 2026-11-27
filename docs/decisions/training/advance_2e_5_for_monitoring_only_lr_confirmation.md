# Advance `2e-5` for monitoring-only LR confirmation

Date: 2026-09-04.
Status: historical; governs no future work.

- The five-cell seed-0 Qwen3.5-4B screen selected AdamW `2e-5` over the freshly
  retrained `5e-5` control on the frozen 3,012-row ID development suite. Macro
  pAUROC@20 improved from `0.847144` to `0.871066`, macro AUROC improved from
  `0.951173` to `0.957643`, and macro Brier fell from `0.095650` to `0.079273`.
- The selected rate gained `0.051967` pAUROC@20 on Gloom and lost `0.004124` on
  STRIDE, satisfying the predeclared per-source guardrail. AdamW `1e-4` was also
  eligible but ranked behind `2e-5`; `1e-5` narrowly failed the STRIDE guardrail
  and `2e-4` was rejected.
- Advance `2e-5` to replicated confirmation against `5e-5`. Treat it only as a
  monitoring-only candidate until the seed comparison is complete; the strict
  OOD suite remains unconsulted and must not be used to tune this choice.
