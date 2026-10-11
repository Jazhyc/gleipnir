# Strict-filtering OOD follow-up

User-requested transfer evaluation of the fixed final seed-0 strict-filtered
checkpoint from the [ID sweep](../teacher_disagreement_filtering/README.md).
Hypothesis: its ID ranking gain transfers to the frozen six-source OOD suite.
No further filtering delta, checkpoint, prompt or threshold selection on OOD.
No training, new teacher calls or model promotion.

Score all 6,395 unchanged examples with the compact training instruction and
BF16 vLLM 0.31 causal LAST two-logit scoring, c128 and 128-row groups. Reconstruct
the candidate in the existing shared ephemeral merge slot, using FP32 merge
accumulation and BF16 export. Verify every merged file against the candidate's
completed ID merge. Reuse its checksum-bound master/merged reference receipt
only after exact identity checks; perform a fresh serving canary and native
BF16 audit before any OOD scoring. Keep persistent caches and FP32 adapters.
This continues the user's BF16 exception to optimized serving.

The user requests **cached Gleipnir 4B scores**, not a fresh control run. The
primary reference is the released mixed-data 4B model's saved full-suite BF16
predictions. Also report the cached regular, unfiltered SDPA model for context.
The released model includes 13,149 additional deception examples; the regular
model uses the same original monitoring pool as the filtering sweep. Both cached
references used historical dynamic-LoRA serving/runtime/host/batching. These
comparisons do not isolate filtering from all training/serving differences.

Primary metric: equal-source raw normalized pAUROC@20. Report pooled/source
AUROC, Brier, calibration, threshold-0.5 FPR/recall and ties. Audit IDs, source,
labels and exact source/rendered prompt identities against cached predictions.
Require complete finite raw scores and no truncation. Disclose prior repeated
use of this OOD suite; this is a transfer diagnostic on an existing benchmark.

Stop on checksum/source/runtime/GPU drift, a differing reconstructed merge,
failed serving parity/native checks, backend error or incomplete membership.
Use a detached runner with a one-hour serving-stage deadline; no automatic
retraining, precision fallback or failed-parity waiver. The Pod stays running.
The prior user preference for unattended execution applies; no agent scheduler
is available. Preserve all outputs and failed receipts on the workspace volume.

`prepare.py` freezes inputs, cached references, candidate receipts and rendered
workloads locally. `run.py` reuses the shared monitoring merge/scoring helpers
with an OOD context, retires the scorer on completion and verifies full metrics.
Results: `results/teacher-disagreement-strict-ood01/`; data:
`data/teacher-disagreement-strict-ood01/`; logs:
`logs/runpod/teacher-disagreement-strict-ood01/`.

```bash
PYTHONPATH=src:. python -m experiments.teacher_disagreement_ood.prepare
PYTHONPATH=src:. python -m experiments.teacher_disagreement_ood.run
```
