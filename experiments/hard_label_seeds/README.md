# Hard-label 4B seeds matched to the soft-only release seeds

User request, 2026-10-24: give the probe study a matched LoRA reference for the
objective question. The three soft-only seeds of the strict-filtered
BF16/SDPA 4B (`teacher-disagreement-delta000-sdpa04`, `teacher-seed-sdpa-s1-01`,
`teacher-seed-sdpa-s2-01`; OOD 0.787 / 0.813 / 0.777) have no hard-only
counterpart on the trajectory data; the earlier hard-versus-soft OOD comparison
was on the Aletheia's Quest deception data, and the trajectory hard-label
screen was ID-only with one seed.

- **Hypothesis.** Kimi K3 soft targets give the 4B LoRA monitor higher OOD
  pAUROC@20 than source hard labels alone.
- **Intervention.** Seeds 0, 1 and 2 copy the seed-0 campaign and change only
  the seed, the objective (`soft_loss_weight` 0, `direct_loss_weight` 1: the
  decision token trains on the source label and no Kimi artifact reaches the
  trainer), identity, GPU, merge slot, serving hash and contract text. Data
  (7,836 strict-filtered rows), initialization, LR 5e-5, one epoch of 245
  updates, rank and the packed BF16/SDPA profile are unchanged; the seed sets
  data order exactly as for the soft seeds. `test_contract.py` enforces this.
- **Scoring.** Each campaign scores the 3,012-row CoT-removed ID set; a
  separate context scores the frozen 6,395-row OOD suite once, with BF16 vLLM
  0.31 at c128, as for the soft seeds.
- **Selection.** Final checkpoint only; descriptive 3-vs-3 comparison, no
  promotion. Report seed means, per-source ranking and calibration.
- **Stop** on input or runtime drift, nonfinite or missing gradients, failed
  master/merge/serving parity, incomplete scoring, or completion.
- **Compute.** One EU-RO-1 B200 Pod on volume `cb7bxu71ug`, authorized by the
  user for these three seeds and to be terminated after scoring and artifact
  collection. The runtime is restored as in
  `logs/runpod/distillation_temperature/restore.sh`.

```bash
PYTHONPATH=src:. <training-python> -m experiments.hard_label_seeds.prepare --gpu-uuid UUID
for s in 0 1 2; do PYTHONPATH=src:. <training-python> -m gleipnir.campaigns.monitoring \
  prepare --config data/teacher-hard-sdpa-s$s-01/config.yaml; done
PYTHONPATH=src:. <training-python> -m experiments.hard_label_seeds.run
```
