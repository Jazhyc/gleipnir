# Hard-label Qwen3.8-27B matched to the soft 27B seed 0

User request, 2026-11-03: give the 27B a hard-label LoRA baseline. The
[first 27B campaign](../qwen38_27b_campaign/README.md) trained on Kimi K3
soft targets only (ID 0.9217, OOD 0.9087). Every hard-label LoRA so far is
4B ([hard-label seeds](../hard_label_seeds/README.md): soft targets add
+0.069 ID and +0.076 OOD there).

- **Hypothesis.** Kimi K3 soft targets give the 27B LoRA monitor higher ID
  and OOD pAUROC@20 than source hard labels alone, as they do for the 4B.
- **Intervention.** `qwen38-27b-hard-s0-01` copies `qwen38-27b-s0-01` and
  changes only the objective (`soft_loss_weight` 0, `direct_loss_weight` 1:
  the decision token trains on the source label and no Kimi artifact reaches
  the trainer), identity, GPU, merge slot, serving hash and contract text.
  Data (7,836 strict-filtered rows), seed 0, the 27B initial adapter, LR 5e-5,
  one epoch of 245 updates and `qwen38_27b_b200_default` are unchanged.
  The runner still requires full startup validation for 27B profiles.
  `test_contract.py` enforces this.
- **Scoring.** The campaign scores the 3,012-row CoT-removed ID set; the
  context `qwen38-27b-hard-s0-ood01` scores the frozen 6,395-row OOD suite
  once, with BF16 vLLM 0.31 at c128, as for the soft 27B.
- **Selection.** Final checkpoint only; one seed, so the comparison with the
  soft 27B is descriptive (the 4B OOD seed SD is 1.8 points). No promotion.
- **Stop** on input or runtime drift, failed startup gates, nonfinite or
  missing gradients, OOM, failed master/merge/serving parity, incomplete
  scoring, or completion.
- **Compute.** One EU-RO-1 B200 Pod on volume `cb7bxu71ug`, authorized by the
  user for this run; kept running after scoring at the user's request for
  follow-ups.

```bash
PYTHONPATH=src:. <training-python> -m experiments.qwen38_27b_hard_labels.prepare --gpu-uuid UUID
PYTHONPATH=src:. <training-python> -m gleipnir.campaigns.monitoring \
  prepare --config data/qwen38-27b-hard-s0-01/config.yaml
PYTHONPATH=src:. <training-python> -m experiments.qwen38_27b_hard_labels.run
```

## Run log (2026-11-03, EU-RO-1 B200 `GPU-2ab4adcc…`, driver 595.91.07)

Pod `6ucjc9wunbydk8` on volume `cb7bxu71ug`. The volume returned `ENXIO`
for 55 files (list: `logs/runpod/qwen38_27b_hard_labels/volume_scan_bad.txt`;
see [infrastructure](../../docs/infrastructure.md)), so the runtime was rebuilt
on the container disk: `.venv-vllm031` by `uv sync --locked` (staged metadata
hashes identical to the previous receipt), FlashQLA from fresh TileLang 0.1.12 /
TVM-FFI 0.1.11 wheels plus the recorded FP16 patch (all 43 manifest hashes
verified), FLA without `__pycache__`, and cold compiler caches. Damaged volume
copies are kept as `*.damaged-20261103`.

All 119 bound inputs verified at preparation. Full startup validation passed
(packing eager/compiled canaries, adaptive gradient canary relative L2 0.016,
longest-32 preflight, trim canary). Training ran 17:24–21:28 UTC, 245 updates
at about 60 s each; loss fell from 0.11 (updates 1–10) to about 1e-5 by the
end. Merged parity: correlation 0.9999999997, MAE 4.1e-6, adapter effect
0.95. ID (3,012 rows) and OOD (6,395 rows) scored with BF16 vLLM 0.31; master
`f7e9aad6…`.

## Result (`compare.py`, `results/qwen38_27b_hard_labels/comparison.json`)

Source-macro metrics on identical rows; "extreme" is the share of scores
within 0.01 of 0 or 1.

| | ID pAUROC@20 | ID Brier | OOD pAUROC@20 | OOD AUROC | OOD Brier | OOD extreme |
|---|---:|---:|---:|---:|---:|---:|
| **27B hard, seed 0** | **0.9665** | 0.0723 | 0.9044 | 0.9655 | 0.0712 | 0.79 |
| 27B soft, seed 0 | 0.9217 | 0.0663 | **0.9087** | **0.9679** | **0.0574** | 0.17 |
| 27B unadapted | 0.8869 | — | 0.8901 | — | — | — |
| 4B hard, seed mean | 0.8334 | 0.1310 | 0.7164 | 0.8833 | 0.2686 | 0.82 |
| 4B soft, seed mean | 0.9026 | 0.0739 | 0.7921 | 0.9179 | 0.1391 | 0.01 |

| OOD source | 27B hard | 27B soft |
|---|---:|---:|
| AgentDojo | 0.9987 | 0.9983 |
| CUA-SHADE-Arena | 0.8775 | 0.8806 |
| IaC | 0.9428 | 0.9871 |
| MLE-Sabotage | 1.0000 | 1.0000 |
| Rogue Eval | 0.9976 | 0.9989 |
| SHADE-Arena | 0.6100 | 0.5875 |

- **ID favors hard labels at 27B** (+4.5 points; Gloom 0.944 vs 0.887,
  STRIDE 0.989 vs 0.957), the opposite of the 4B (−6.9). ID labels are the
  source labels the hard arm fits directly.
- **OOD is a near tie** (−0.4 points, within the 4B OOD seed SD of 1.8): hard
  loses 4.4 on IaC and gains 2.3 on SHADE-Arena. Both beat the unadapted 27B
  (+1.4 hard, +1.9 soft). At 4B, soft wins OOD by 7.6.
- **Soft keeps the calibration advantage**: OOD Brier 0.057 vs 0.071, and
  hard scores saturate (79% extreme vs 17%).
- Qualifications: one seed per arm; hard labels trained with the untuned soft
  recipe; the unadapted 27B was scored under the teacher prompt.
