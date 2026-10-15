# Seed variation of the strict-filtered SDPA teacher

First step of the teacher-precision study. The goal is to separate the MLP
(FP4 vs BF16) and attention (FA4 vs SDPA) contributions to the FP4/FA4
teacher's OOD loss. Only the two diagonal cells exist, each with one seed.
Before attributing a difference between cells, measure how much one teacher
varies across seeds.

**Hypothesis.** Retraining the strict-filtered BF16/SDPA teacher
(`teacher-disagreement-delta000-sdpa04`, seed 0) with seed 1 changes OOD and
ID ranking much less than the FP4/FA4 gap (OOD −4.90 pAUROC@20 points, ID
−0.89).

**Intervention.** `teacher-seed-sdpa-s1-01` copies the seed-0 campaign config
and changes only the seed, identity, GPU, merge slot and contract text. The
seed controls data order and packing. Initialization (`a6b1d2e9…`), data,
Kimi targets, profile, schedule and packing budget are unchanged; a test
enforces this. The serving-selection hash is updated because the serving
default changed after seed 0. BF16 scoring replaces the optimized workers.

**Scoring.** All runs use BF16 vLLM 0.31 at c128. Seed 1 is scored on ID
(3,012 rows) and the frozen OOD suite (6,395 rows). Seed 0 is rescored on
both in the same serving session, as a drift control against its cached
scores.

**Selection.** Descriptive only; no checkpoint is promoted. Under the
[precision-selection decision](../../docs/decisions/precision_selection_on_ood.md),
later precision cells may select on OOD. This run sets the noise floor for
that.

**Runtime.** Training runs on the EU-RO-1 B200 with the NC2 training runtime
restored from `container_runtimes.tar.zst` (SHA-256 `c4d76a28…`).
- The runtime sits at `/tmp/gleipnir-training-runtime`, because the EU serving
  runtime keeps `/tmp/gleipnir-serving-runtime` (its FA4/MXFP8 overlays differ).
- FLA, causal-conv1d and Triton link to `.cache/kernels-nc2/`; FlashQLA is at
  `.cache/kernels/flashqla-da06429`.
- Versions match the seed-0 receipt, and all 43 FlashQLA files match their
  hashes.
- The EU `.cache/kernels/fa4` overlay has identical TVM-FFI code to NC2's; it
  lacks NC2's CUDA-12 CUTLASS libraries, `quack` and `torch_c_dlpack_ext`,
  which SDPA training does not use. The original
seed-0 run used a different host and GPU, so seed and host effects are
confounded at the seed-0/seed-1 boundary. Same-host repeatability of this
recipe is bitwise for BF16/FA4 (augmented line) and assumed, not measured,
for SDPA.

**Stop** on input or runtime drift, nonfinite or missing gradients, failed
master/merge/serving parity, incomplete scoring, or completion.

```bash
PYTHONPATH=src:. <training-python> -m experiments.teacher_seed_variation.prepare --gpu-uuid UUID
PYTHONPATH=src:. <training-python> -m gleipnir.campaigns.monitoring prepare \
  --config data/teacher-seed-sdpa-s1-01/config.yaml
PYTHONPATH=src:. <training-python> -m experiments.teacher_seed_variation.run
python -m experiments.teacher_seed_variation.compare
```

## Results (2026-10-15/16, EU-RO-1 B200 `GPU-e5eb9962…`)

Seed 1 trained all 245 updates (about 11–12 s per update once compiled). It
passed master, merge and BF16 serving parity (merged correlation 0.99998,
adapter effect 0.85).

The seed-0 rescore re-merged bitwise-identical weights. Its drift against the
cached Oct 11/12 scores is under 0.002 points on ID and OOD (mean score shift
0.0013–0.0017, 3 and 9 threshold flips). That matches the FP4/FA4
serving-repeat floor (0.015 points, mean shift 0.0015), so serving is not a
confound.

Source-macro pAUROC@20 (`results/teacher_seed_variation/comparison.json`):

| Teacher, BF16 serving | ID | OOD |
| --- | ---: | ---: |
| BF16/SDPA seed 0 (rescored) | 0.8959 | 0.7869 |
| BF16/SDPA seed 1 | 0.8978 | 0.8125 |
| FP4/FA4 seed 0 | 0.8870 | 0.7380 |

Differences in points, by source:

| Comparison | ID macro | OOD macro | OOD AgentDojo | CUA-SHADE | IAC | Rogue eval | SHADE-Arena |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed 1 − seed 0 | +0.19 | **+2.56** | −0.50 | −0.77 | +8.35 | +1.15 | +7.12 |
| FP4/FA4 − SDPA seed 0 | −0.89 | **−4.89** | −3.79 | −3.51 | −8.27 | −7.81 | −5.98 |

MLE sabotage is saturated (1.0) in every run. Between seeds, 340/3,012 ID and
497/6,395 OOD scores cross 0.5 (mean shifts 0.096 and 0.077).

**What this means.**
- **Seed effect.** Data order alone moves OOD macro pAUROC by 2.6 points, about
  half the FP4/FA4 gap, and moves IAC and SHADE-Arena by about 7–8 points each.
  On ID, the per-source seed swing (±1.2–1.6) exceeds the whole FP4/FA4 gap.
- **The FP4/FA4 gap.** Its pattern differs from the seed effect: all five
  unsaturated OOD sources are worse (−3.5 to −8.3), while the seed change is
  mixed in sign. Rogue eval (−7.8 vs ±1.2) and AgentDojo/CUA-SHADE (−3.5 to −3.8
  vs under ±0.8) fall outside the single observed seed swing.
- **Not established.** With one seed difference, neither the seed spread nor
  the size of an FP4/FA4 effect is established. The factorial cells need
  multiple seeds, or a per-source analysis against a measured seed
  distribution, before attributing OOD differences to MLP precision or
  attention.

**Failure and recovery.** The OOD-only seed-1 scoring context inherited
`evaluations: [id]`, so the shared summary raised `KeyError: 'id'` after all
6,395 rows were scored and parity-gated. The predictions were kept with
`results/teacher-seed-sdpa-s1-ood01/recovery.json`. The seed-0 rescore ran
afterwards with user approval, with unchanged hash-bound sources.
`scoring_config` now sets evaluations from the scored splits, for future runs.

## Three-seed spread (seed 2 added 2026-10-16)

Seed 2 trained 245 updates in 2,886 s with a warm compile cache. It passed
master, merge and serving parity, and was scored on ID and OOD (`status_s2`).
Source-macro and per-source pAUROC@20 for BF16/SDPA seeds 0/1/2, with the
FP4/FA4 seed-0 teacher (`comparison.json` → `seed_spread`):

| Split / source | Seed 0 | Seed 1 | Seed 2 | Mean | SD (pt) | Range (pt) | FP4/FA4 | FP4/FA4 − lowest seed (pt) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID macro | 0.8959 | 0.8978 | 0.9142 | 0.9026 | 1.01 | 1.83 | 0.8870 | −0.89 |
| ID Gloom | 0.8285 | 0.8168 | 0.8619 | 0.8357 | 2.34 | 4.51 | 0.8189 | +0.21 |
| ID STRIDE | 0.9633 | 0.9788 | 0.9665 | 0.9696 | 0.82 | 1.55 | 0.9552 | −0.81 |
| OOD macro | 0.7869 | 0.8125 | 0.7768 | 0.7921 | 1.84 | 3.57 | 0.7380 | **−3.87** |
| AgentDojo | 0.9298 | 0.9248 | 0.8506 | 0.9017 | 4.44 | 7.92 | 0.8919 | +4.13 |
| CUA-SHADE | 0.7717 | 0.7640 | 0.7679 | 0.7679 | 0.39 | 0.77 | 0.7366 | **−2.74** |
| IAC | 0.8123 | 0.8958 | 0.8451 | 0.8511 | 4.21 | 8.35 | 0.7296 | **−8.27** |
| Rogue eval | 0.9696 | 0.9812 | 0.9765 | 0.9758 | 0.58 | 1.15 | 0.8916 | **−7.81** |
| SHADE-Arena | 0.2380 | 0.3092 | 0.2209 | 0.2561 | 4.68 | 8.83 | 0.1783 | **−4.26** |
| MLE sabotage | 1.0000 | 1.0000 | 0.9995 | 0.9998 | 0.03 | 0.05 | 1.0000 | +0.05 |

What this shows:
- **SDPA seed noise is large and source-dependent.** OOD macro SD is 1.8
  points (range 3.6). AgentDojo, IAC and SHADE-Arena have SDs of 4–5 points;
  CUA-SHADE and Rogue eval stay under 0.6.
- **ID and OOD do not move together across seeds.** Seed 2 has the best ID
  macro (+1.8) and the worst OOD macro (−1.0 versus seed 0).
- **The FP4/FA4 OOD deficit survives the seed noise.** FP4/FA4 is below every
  SDPA seed on OOD macro (5.4 points under the mean, about 2.9 SDs) and on
  CUA-SHADE, IAC, Rogue eval and SHADE-Arena. On the low-noise sources
  (CUA-SHADE, Rogue eval) the gap is 5–14 SDs. AgentDojo is within the
  seed range.
- **The ID deficit is marginal.** ID macro is 0.89 points below the lowest
  SDPA seed, and Gloom is inside the range.
- **Limit.** FP4/FA4 has one seed, so its own spread is unmeasured. The
  attribution to MLP precision versus attention still needs the off-diagonal
  cells.
