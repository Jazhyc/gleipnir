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

**Run log.** Seed 0 completed training, ID and OOD scoring. Seed 1 trained all
245 updates but its final checkpoint save failed: the 200 GB network volume
hit its quota ("Disk quota exceeded"). Existing volume data was left
untouched; the failed seed-1 partial output was removed, the `4b/` training
output directories of seeds 1 and 2 were redirected to the Pod's container
disk, and the queue resumed with `run.py --resume` (seed 0 skipped). Editing
this experiment's sources tripped the campaign source-drift check, so seeds 1
and 2 were re-frozen (`prepare.py --seeds 1 2`) against the current sources;
seed 0 keeps its original contract. Stale prepared state was moved aside.
Everything is pulled before the Pod is terminated.

## Result (2026-10-24/25, EU-RO-1 B200 `GPU-338d6756…`)

All three seeds trained 245 updates (about 11.7 s per update), passed
master-to-merge parity (MAE ≤ 7.2e-5) and were scored on ID and the frozen OOD
suite with BF16 vLLM 0.31. Soft-only values are the existing seeds (seed 0
rescored in the seed-variation session).

| seed | hard ID | soft ID | hard OOD | soft OOD |
|---|---:|---:|---:|---:|
| 0 | 0.8100 | 0.8959 | 0.7458 | 0.7869 |
| 1 | 0.8669 | 0.8978 | 0.7131 | 0.8125 |
| 2 | 0.8231 | 0.9142 | 0.6902 | 0.7768 |
| mean | 0.8333 | 0.9026 | 0.7164 | 0.7921 |

Removing Kimi soft targets costs −0.069 ID and −0.076 OOD (macro pAUROC@20);
every hard seed is below every soft seed on both. ID losses are on Gloom
(0.63–0.76 versus 0.82–0.86); STRIDE is slightly higher with hard labels. OOD
losses are largest on IaC (0.55–0.70 versus 0.81–0.90) and CUA-SHADE. Hard
labels also calibrate far worse (OOD Brier 0.21–0.30 versus 0.14) and saturate
scores (seed 0 has 197 distinct Gloom scores). Hard seeds vary more on ID
(range 0.057 versus 0.018), and ID rank does not predict OOD rank. For
comparison, soft targets help activation probes by +0.007 (prompted) and
+0.055 (NTP) OOD at the selected cells (`experiments/probe_generalization/`).
Artifacts: `results/teacher-hard-sdpa-s{0,1,2}-01`, `…-ood01`,
`results/hard_label_seeds/` (status, stale prepared state) and the failed
seed-1 receipt `results/teacher-hard-sdpa-s1-01_failed_quota01`; FP32 masters
are also on volume `cb7bxu71ug`. The Pod was terminated at the user's
instruction after all files were verified locally (2,991 files by size; 9
safetensors by SHA-256).

