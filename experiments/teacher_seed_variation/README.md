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
