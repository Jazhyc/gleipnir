# Final two-logit score resolution

Hypothesis: avoiding BF16 rounding in the final two-row classifier recovers
useful ordering within tied scores without changing the monitor backbone.
Use the fixed final strict delta-zero seed-0 checkpoint, its existing checksum-
verified BF16 merge and archived BF16 predictions. No training, new annotation,
checkpoint/threshold selection, model promotion or capacity lifecycle change.

Keep all backbone projections, hidden states, attention, cache, GDN kernels,
tokenizer, causal LAST pooling, grouping 128 and concurrency 128 identical to
the completed strict ID/OOD run. Promote the same two BF16 weight rows and LAST
hidden states to FP32 for the final dot products only. Disable TF32 and retain
FP32 head output through the response; sigmoid already uses Python floats.
The stored model weights remain unchanged. Verify actual input/weight/output
dtypes and weight identity; the parent BF16 worker continues to audit the
backbone and native attention. Require the original adapter-specific serving
canary before scoring all 3,012 ID and 6,395 OOD rows. Never waive failed parity.

Baseline: saved strict-model BF16 ID/OOD predictions and matching timing files,
on the same GPU/runtime/merge. Report ties, score/margin/threshold differences,
source-macro and pooled pAUROC@20/AUROC, calibration, source metrics, input
tokens/s, requests/s and latency. One full pass is a precision diagnostic;
single-pass throughput differences include timing variation. OOD has prior
use; this test does not select a new recipe on OOD. Stop on drift, failed dtype
or canary checks, nonfinite scores, incomplete coverage, server failure or the
one-hour scoring deadline. Retire the owned scorer on completion/failure;
preserve caches, original predictions, adapters and shared merged checkpoint.

`prepare.py` freezes new source/input identities without modifying previous
campaigns. `run.py` verifies the existing merge, reuses its identity-bound master
reference, and uses the shared scorer with an experiment-local worker override.
Data/results/logs use `fp32-monitor-head01`. Run on the existing authorized NC2
B200, with no new capacity or lifecycle action. No scheduler is available;
the detached driver provides automation, with startup checked in the active turn.

```bash
PYTHONPATH=src:. python -m experiments.monitor_score_resolution.prepare
PYTHONPATH=src:. python -m experiments.monitor_score_resolution.run
```
