# QAD for Gleipnir 4B v0.2

Gleipnir 4B v0.2 loses 5.3 Mean-OOD pAUROC@20 points on the optimized serving
stack (0.8108 BF16 against 0.7579;
[optimized gap](../optimized_gap/README.md)). The 27B loses nothing and
needs no QAD ([decision](../../docs/decisions.md)).

- **Hypothesis.** Same-model quantization-aware distillation from v0.2's own
  BF16 scores recovers a substantial part of that optimized-serving loss, on
  ID and OOD.
- **What is new.** This is the first QAD from a BF16/SDPA-trained model. The
  original recovery ([finding](../../docs/findings/quantization_recovery.md))
  started from the FP4/FA4-trained baseline. It recovered 71% of the ID gap,
  but not the OOD gap ([QAD OOD](../../docs/findings/qad_ood.md)).
- **Recipe.** It is unchanged from the original QAD: the 7,836 strict-filtered
  rows (the same training file and prompt bytes as v0.2), KL at T=1, AdamW
  1e-5, no weight decay, 3% warmup with linear decay, 245 updates, logical
  batch 32, 16K packing, 29,696 context. It uses the
  [selected lean QAT arithmetic](../../docs/decisions/qat_reference_training.md):
  native NVFP4 MLP/GDN forward, decoded FP8 projection reference, serving
  MXFP8 attention forward, quantized-QKV FA4 surrogate backward, BF16
  FlashQLA and FP32 masters.
- **Student.** The exact rank-384 v0.2 FP32 adapter. The QAD arithmetic is
  rank-agnostic: the straight-through backward reads each module's A, B and
  scaling. The rank-128 check belongs to a different recipe's identity gate.
- **Teacher.** v0.2's own BF16 scores on the training rows, from the v0.2
  merge built and parity-checked on this host (`teacher.py`).
- **Gates.** The original preflight runs unchanged:
  - validated operator and decoder receipts;
  - simulator against the *current* optimized server on the 20 v0.2 canary
    prompts (mean difference ≤ 0.02, correlation ≥ 0.99);
  - packed-sequence isolation;
  - longest-row finite gradients and memory.

  `generate.py` changes only the config, cache, runtime hooks, canary
  reference and starting-server canary in the frozen stages, with
  substitution receipts.
- **Evaluation.** First v0.2's optimized ID baseline, then the QAD master in
  campaign layout through the `optimized_gap` path: fresh merge on this host,
  master-to-merged parity, optimized parity gate, and ID (3,012 rows) and OOD
  (6,395 rows) scoring. The comparisons are v0.2 BF16 and v0.2 optimized
  without QAD. Final epoch only; nothing is selected or promoted.

```bash
PYTHONPATH=src:. <training-python> -m experiments.qad_v02.run
```

## Run (2026-10-20, EU-RO-1 B200 `GPU-3d8d6bde…`)

- **Baseline.** v0.2 on the optimized stack scores 0.8807 ID macro
  pAUROC@20, against 0.9124 BF16 (−3.2 points; Gloom 0.8517 → 0.7897, STRIDE
  0.9732 → 0.9717). Mean score change 0.053, 157 threshold flips.
- **Startup.**
  - Attempt 1 failed at import: the EU serving FA4 overlay lacks the bundled
    `quack` 0.6.5, so the venv's DSL-4.5-era `quack` failed under CUTLASS DSL
    4.8 (`cute.core` has no `ThrMma`). The training and reference stages now
    use the NC2 runtime's FA4 overlay (`/tmp/gleipnir-training-runtime/fa4`).
    It is the overlay the original QAD trained with, and the change is
    recorded in the execution contract.
  - Attempts 2 and 3 stopped at orchestration guards before any training.
  - All receipts are preserved under `results/qad-v02-01_failed_*`.
- **Preflight (attempt 4).** All gates passed:
  - simulator against the current optimized server on the 20 v0.2 canary
    prompts: mean difference 0.0048, correlation 0.99984, maximum 0.025;
  - packed isolation: 0.0;
  - longest row (29,337 tokens): finite gradients (norm 31.8), peak 129.4 GiB.
    Rank 384 adds about 1.3 GiB over the original rank-128 run.
- **Training.** 245 updates in 64.3 minutes. Every loss and gradient was
  finite, with KL following the original QAD trajectory (same seed and batch
  order). One isolated gradient-norm spike reached 8.2 (the original peaked
  at 3.0). FP32 master `0716efc1…`.
- **Packaging stall.** The driver died silently while copying the
  campaign-layout package onto the EU network volume, which had hit its
  200 GB quota (a later write returned `Disk quota exceeded`). Training and
  reference receipts were complete. The verified-identical duplicate master
  copy and the partial package (about 4 GB) were removed; this is recorded in
  `resume_from_package.json`. `run.py --resume-package` rebuilt the package on
  container disk, linked from `results/qad-v02-01/4b`, and evaluated.
- **Gates.** Merged parity: mean difference 0.0030, correlation 0.99992.
  Optimized canary gate: 0.0095 and 0.99946. MXFP8 compiled on the first
  attempt.

## Result: ID fully recovered, OOD mostly not

| Macro pAUROC@20 | v0.2 BF16 | v0.2 optimized | QAD, optimized | Gap recovered |
|---|---:|---:|---:|---:|
| ID (3,012) | 0.9124 | 0.8807 | **0.9117** | **97.6%** |
| OOD (6,395) | 0.8108 | 0.7579 | **0.7703** | **23.4%** |

| Other metrics | v0.2 BF16 | v0.2 optimized | QAD, optimized |
|---|---:|---:|---:|
| ID AUROC / Brier | 0.9702 / 0.0622 | 0.9571 / 0.0740 | 0.9711 / 0.0615 |
| OOD AUROC / Brier | 0.9199 / 0.1339 | 0.9006 / 0.1557 | 0.9138 / 0.1579 |
| OOD pooled pAUROC@20 | 0.7514 | 0.6570 | 0.7005 |

OOD pAUROC@20 by source (BF16 → optimized → QAD optimized):

| Source | BF16 | Optimized | QAD optimized |
|---|---:|---:|---:|
| AgentDojo | 0.9430 | 0.7913 | 0.8401 |
| SHADE-Arena | 0.2653 | 0.1852 | 0.2524 |
| Rogue Eval | 0.9804 | 0.9567 | 0.9586 |
| MLE-Sabotage | 1.0000 | 0.9990 | 1.0000 |
| IaC | 0.8845 | 0.8347 | 0.8214 |
| CUA-SHADE | 0.7916 | 0.7805 | 0.7493 |

- **ID.** QAD from the BF16/SDPA model closes the optimized-serving ID gap
  almost completely: Gloom 0.7897 → 0.8513 against 0.8517 in BF16. The
  FP4/FA4-based QAD recovered 71%.
- **OOD.** The gap only partly closes: −5.3 → −4.05 points.
  - AgentDojo recovers about a third (+4.9) and SHADE-Arena most of its loss
    (+6.7).
  - IaC (−1.3) and CUA-SHADE (−3.1) get worse than optimized serving without
    QAD.
  - Pooled pAUROC@20 recovers more (0.657 → 0.701); OOD Brier gets slightly
    worse.
- **Scores.** QAD-optimized scores still differ from BF16 by 0.052 (ID) and
  0.060 (OOD) on average, with 152 and 438 threshold flips. Matching the
  teacher's ranking on the training distribution did not make per-example
  scores match off-distribution.
- **Reading.** Same-model QAD fits the quantized model to its BF16 self where
  the distillation data is; the OOD drift lies outside it. That fits the
  original QAD's ID-only recovery and points at the distillation data
  distribution, not at the arithmetic.
