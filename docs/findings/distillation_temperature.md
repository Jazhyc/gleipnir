# Distillation temperature

Status: complete, 2026-10-16. ID screen, one seed, no promotion. See the
[experiment](../../experiments/distillation_temperature/README.md) for the
audit, design and per-source results.

**Teacher targets.** The strict-filtered Kimi K3 targets are bimodal: 55% lie
outside 0.01–0.99. They are not saturated: margins span ±12 on a 1/8 grid.
Seed 0–2 students produce ID margins about half as large as the teacher's
training margins.

**Results.** All three arms change only the soft loss of the seed-1
BF16/SDPA recipe. Every one lowers held-out ID macro pAUROC@20 against the
seed-matched control (0.8978):
- KD T=2: −1.9 points;
- KD T=4: −1.8 points, mainly STRIDE;
- teacher-only sharpening (logit ×2): −6.9 points, with Gloom −13.6.

Higher T widened the student's margins as intended but did not improve ranking.

Keep the T = 1 binary soft BCE. Further temperature work has low priority
unless a different objective changes the picture. The trainer keeps
`soft_kd_temperature` (default 1) for that case.
