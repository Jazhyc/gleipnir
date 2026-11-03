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
  user for this run, terminated after scoring and artifact collection.

```bash
PYTHONPATH=src:. <training-python> -m experiments.qwen38_27b_hard_labels.prepare --gpu-uuid UUID
PYTHONPATH=src:. <training-python> -m gleipnir.campaigns.monitoring \
  prepare --config data/qwen38-27b-hard-s0-01/config.yaml
PYTHONPATH=src:. <training-python> -m experiments.qwen38_27b_hard_labels.run
```
