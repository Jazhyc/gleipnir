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
