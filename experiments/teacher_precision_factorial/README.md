# Teacher precision factorial (MLP precision × attention)

This separates the MLP (BF16 vs native FP4) and attention (SDPA vs FA4)
contributions to the FP4/FA4 teacher's OOD loss. The existing diagonal cells
are BF16/SDPA (seeds 0–2, [seed variation](../teacher_seed_variation/README.md))
and FP4/FA4 (seed 0). This experiment adds the off-diagonal cells, one seed
(0) each, at the user's request.

**Intervention.** Each cell is the strict-filtered seed-0 SDPA campaign with
only the training profile and its startup-validation reference changed. A
test enforces this.
- **BF16/FA4:** `qwen35_4b_b200_bf16_fa4`, reference
  `b200_bf16_fa4_accepted` (`185fa498…`).
- **FP4/SDPA:** requires a new profile and an FP4 acceptance decision; see
  below.

**Runtime.** The NC2 training runtime is the same as in the seed runs. For
FA4 cells, `.cache/kernels/fa4` points at the NC2 FA4 overlay
(`/tmp/gleipnir-training-runtime/fa4`). Its TVM-FFI, CUTLASS-DSL, FA4 and
metadata hashes match the NC2 overlay the FP4/FA4 teacher trained with. On
the EU volume this path was already a symlink to the EU serving runtime's
overlay (`/tmp/gleipnir-serving-runtime/fa4`), which stays untouched. Point
the link back there before restaging EU serving. Serving reads its staged
copy either way.

**Scoring and selection.** BF16 serving on ID (through the runner) and on the
frozen OOD suite. Comparisons are descriptive against the three-seed SDPA
spread. Under the
[precision-selection decision](../../docs/decisions/precision_selection_on_ood.md),
OOD may inform precision choices. With one seed per cell, attribution relies
on the low-noise OOD sources (CUA-SHADE, Rogue eval) and OOD macro, relative
to the SDPA seed SD.

**Stop** on input or runtime drift, nonfinite or missing gradients, failed
parity, incomplete scoring, or completion.

## BF16/FA4 result (`teacher-cell-bf16-fa4-s0-01`, 2026-10-16)

Training: 245 updates in 2,248 s, train loss 0.1869. Startup reuse of the
accepted BF16/FA4 receipt passed, as did master, merge and BF16 serving parity
(OOD serving parity passed). pAUROC@20 against the three SDPA seeds
(`results/teacher_precision_factorial/bf16_fa4_comparison.json`):

| Split / source | SDPA mean (SD) | BF16/FA4 | FP4/FA4 | FA4 effect (BF16/FA4 − SDPA mean) | FP4 effect at FA4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ID macro | 0.9026 (1.01) | 0.8605 | 0.8870 | −4.21 | +2.65 |
| ID Gloom | 0.8357 (2.34) | 0.7546 | 0.8189 | −8.11 | +6.42 |
| ID STRIDE | 0.9696 (0.82) | 0.9664 | 0.9552 | −0.32 | −1.11 |
| OOD macro | 0.7921 (1.84) | 0.7579 | 0.7380 | −3.42 | −1.99 |
| AgentDojo | 0.9017 (4.44) | 0.7549 | 0.8919 | −14.68 | +13.70 |
| CUA-SHADE | 0.7679 (0.39) | 0.8044 | 0.7366 | +3.66 | −6.78 |
| IAC | 0.8511 (4.21) | 0.7729 | 0.7296 | −7.82 | −4.33 |
| Rogue eval | 0.9758 (0.58) | 0.9342 | 0.8916 | −4.15 | −4.27 |
| SHADE-Arena | 0.2561 (4.68) | 0.2818 | 0.1783 | +2.58 | −10.36 |

Single seed per cell, so the per-source estimates are noisy:
- **FA4 explains most of the OOD macro loss.** BF16/FA4 is below every SDPA
  seed (−3.4 points, about 1.9 SDs). Rogue eval falls −4.2 (about 7 SDs).
- **FA4 also causes a large ID loss, concentrated in Gloom** (−8.1), matching
  the augmented-line SDPA/FA4 finding.
- **FP4 adds further OOD loss under FA4** (−2.0 macro). The low-noise sources
  agree (Rogue eval −4.3, CUA-SHADE −6.8), as do IAC and SHADE-Arena. It
  partly offsets the FA4 ID loss.
- **Large sign changes** (AgentDojo −14.7/+13.7, CUA-SHADE +3.7) show strong
  cell-specific variation or interaction, which one seed cannot resolve.
- **Next.** The FP4/SDPA cell measures the FP4 effect without FA4.

## FP4/SDPA result and full 2×2 (`teacher-cell-fp4-sdpa-s0-01`, 2026-10-16)

Training: 245 updates in 3,308 s (cold FP4 compile), train loss 0.190.
Startup took the composite reuse path
(`reuse_composite_sdpa_packing_and_native_fp4_mlp`) with native FP4 active.
Master, merge and BF16 serving parity passed (OOD serving parity passed).
`results/teacher_precision_factorial/factorial_comparison.json`.

Source-macro pAUROC@20. BF16/SDPA is the three-seed mean (SD); other cells
are seed 0:

| | BF16 MLP, SDPA | FP4 MLP, SDPA | BF16 MLP, FA4 | FP4 MLP, FA4 |
| --- | ---: | ---: | ---: | ---: |
| ID macro | 0.9026 (1.01) | 0.8715 | 0.8605 | 0.8870 |
| ID Gloom | 0.8357 (2.34) | 0.7792 | 0.7546 | 0.8189 |
| ID STRIDE | 0.9696 (0.82) | 0.9638 | 0.9664 | 0.9552 |
| OOD macro | 0.7921 (1.84) | **0.8136** | 0.7579 | 0.7380 |
| AgentDojo | 0.9017 (4.44) | 0.9679 | 0.7549 | 0.8919 |
| CUA-SHADE | 0.7679 (0.39) | 0.7335 | 0.8044 | 0.7366 |
| IAC | 0.8511 (4.21) | 0.8833 | 0.7729 | 0.7296 |
| Rogue eval | 0.9758 (0.58) | 0.9618 | 0.9342 | 0.8916 |
| SHADE-Arena | 0.2561 (4.68) | 0.3353 | 0.2818 | 0.1783 |

Simple effects on OOD macro (points):

| Effect | Change |
| --- | ---: |
| FP4 at SDPA | +2.2 |
| FP4 at FA4 | −2.0 |
| FA4 at BF16 | −3.4 |
| FA4 at FP4 | −7.6 |

What this shows:
- **FA4 is the consistent OOD harm.** It lowers OOD macro under both MLP
  precisions, and Rogue eval falls 4–7 points (7–12 SDs).
- **FP4 alone does not reduce OOD macro.** FP4/SDPA sits at the top of the
  SDPA seed range (+2.2, just above seed 1's 0.8125). The FP4/FA4 deficit
  comes mostly from FA4, plus an FP4×FA4 interaction (FA4 costs 7.6 points
  with FP4 versus 3.4 with BF16).
- **FP4 still costs the low-noise sources.** Under SDPA, CUA-SHADE drops 3.4
  (about 9 SDs) and Rogue eval 1.4 (about 2.4 SDs). The macro gain comes from
  the high-variance AgentDojo, IAC and SHADE-Arena.
- **ID favours BF16/SDPA over every other cell** (−1.6 to −4.2 points), mostly
  through Gloom.
- **Limits.** One seed per off-diagonal and FP4/FA4 cell. Per-source
  differences in the high-variance sources are within or near the SDPA seed
  spread.

**Teacher implication.** Avoid FA4 in teacher training. BF16/SDPA remains the
best-supported teacher: three seeds, best ID, OOD within noise of FP4/SDPA.
FP4/SDPA is a plausible OOD alternative, but its low-noise-source losses and
ID loss would need more seeds before choosing it.

**Pod state.** `.cache/kernels/fa4` and `.cache/kernels/nvidia_mxfp8` still
point at the NC2 training runtime. Repoint them to
`/tmp/gleipnir-serving-runtime/{fa4,mxfp8}` before restaging EU serving.
