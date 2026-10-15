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
