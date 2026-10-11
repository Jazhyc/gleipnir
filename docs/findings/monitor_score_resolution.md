# FP32 final-head computation on the strict-filtered monitor

The final-head precision test removes most BF16 score ties but gives only small
ranking improvements. Source-macro raw normalized pAUROC@20 gains **0.0192
percentage points on ID** and **0.0599 points on OOD** against the saved BF16
strict-model predictions. The same-pass output-rounding comparison also shows
small gains. Final-head rounding is real, but is not a large ranking bottleneck
for this checkpoint. No training or serving-default promotion occurred.

## Fixed intervention

The [experiment contract](../../experiments/monitor_score_resolution/README.md)
fixes the final strict delta-zero seed-0 Qwen3.5-4B checkpoint, its existing
merged BF16 artifact and all canonical ID/OOD prompts. The B200 UUID, vLLM 0.31,
Torch 2.13, tokenizer, causal LAST pooling, BF16 backbone/attention/cache/GDN,
128-row groups and concurrency 128 match the completed strict serving runs.
The base revision is `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
All 33,750,959 ID and 55,346,879 OOD input tokens are processed without truncation.

The worker first runs the existing BF16 weight/native-attention audit, then
copies only the unchanged two decision-token weight rows to FP32. It widens
the BF16 LAST hidden state and computes that two-row projection in FP32,
disabling TF32. Stored weights and all backbone operands remain unchanged.
Actual dispatch proves BF16 input, FP32 compute weights and FP32 output;
all 9,407 responses contain logits outside the original 0.125 grid. Endpoint
normalization already uses Python floats. This is computation before rounding,
not an attempt to recover precision by casting saved logits.

Archived strict BF16 ID/OOD outputs are the primary matched references. A second
counterfactual rounds this pass's FP32 output logits to BF16 before scoring.
That isolates output casting on identical computed logits; it is not another
native BF16 projection or an independent inference baseline. The OOD suite has
prior repeated use. No labels select checkpoint, precision parameters, subset,
threshold, stopping point or a promoted recipe.

## Complete results

Ranking and Brier values below are source-macro; threshold diagnostics retain 0.5.

| Population / head | Unique scores | pAUROC@20 | AUROC | Brier | FPR | Recall |
|---|---:|---:|---:|---:|---:|---:|
| ID / saved BF16 | 98 | 0.895914 | 0.963772 | 0.088954 | 21.45% | 94.37% |
| ID / FP32 | 3,010 | 0.896106 | 0.963869 | 0.088830 | 21.02% | 94.17% |
| OOD / saved BF16 | 80 | 0.786936 | 0.921876 | 0.136523 | 7.72% | 72.04% |
| OOD / FP32 | 6,337 | 0.787535 | 0.922867 | 0.136385 | 7.45% | 71.63% |

Mean absolute probability changes are 0.00405 ID and 0.00407 OOD. There are
13 ID and 26 OOD threshold flips. Lower false-positive rates coexist with
lower recall; Brier improves slightly. Pooled ID pAUROC/AUROC/Brier are
0.855917/0.953576/0.108201; OOD values are 0.710728/0.926747/0.140690.
Full source metrics, calibration, ties and threshold diagnostics are retained.

| Same-pass rounding counterfactual | Rounded pAUROC@20 | FP32 gain | Rounded AUROC | FP32 gain |
|---|---:|---:|---:|---:|
| ID | 0.895929 | +0.000178 | 0.963744 | +0.000125 |
| OOD | 0.787170 | +0.000364 | 0.922243 | +0.000624 |

The same-pass rounded scores differ from archived BF16 by MAE 0.00161 ID and
0.00166 OOD. Therefore the complete archived-reference difference cannot all
be attributed to final output casting. Projection arithmetic and separate-run
serving variation remain qualifications. This single checkpoint/pass does not
establish broad quality improvement, uncertainty calibration equivalence, or
training-seed robustness. It does not explain the large Kimi/student confidence
gap: pooled OOD mean probability is 0.383285, versus 0.383155 previously.

| Population | FP32 scoring seconds | Input tokens/s | Saved BF16 tokens/s | Requests/s | Median request latency |
|---|---:|---:|---:|---:|---:|
| ID | 294.07 | 114,770 | 114,534 | 10.24 | 4.87 s |
| OOD | 491.70 | 112,563 | 112,297 | 13.01 | 3.59 s |

Observed throughput changes are +0.21% and +0.24%. These single-pass differences
are within timing variation; OOD follows a full ID pass in the candidate run.
They do not establish a speed improvement. No meaningful overhead appears in
this measurement, but paired repeated timing would be needed for a tight bound.

## Execution and independent verification

`fp32-monitor-head01` completed at **23:48:30 CEST on 2026-10-11**, in 15m56s
from the validated driver start. Serving canary MAE is 0.00147995, correlation
0.99998136; both FP32-master and BF16-merge gates pass. All eight native BF16
attention calls and the FP32 head dispatch pass. The merge is reused only after
all saved weight/config/tokenizer checksums match. Implementation is commit
`81076e0`; 29 focused tests and lint passed before launch.

Local outputs are under `results/fp32-monitor-head01/` and logs under
`logs/runpod/fp32-monitor-head01/`. `completion_audit.json` independently verifies
all 9,407 raw responses, exact canonical/rendered identities, FP32-representable
logits, original and counterfactual full metric recomputation, dtype/parity
receipts and prediction hashes. ID prediction SHA256 is
`bf6a57aee96d88aa61bf393ba3b7c697c15646c4d8185f4dd814c5cd2ea1ff81`;
OOD SHA256 is
`8ac97421c0e7f7aa0521b7c82de5f3e2db05611cf862a620a4774858166cb37e`.

The scorer is retired; the subsequent GPU check shows 0% utilization and 0 MiB
allocated. The Pod, persistent adapters, caches and shared strict merge remain.
Shutdown logged a force-killed engine worker, an output-handler EngineDeadError
and semaphore cleanup after complete scoring/comparison. There were no scoring
failures or excluded rows; the shutdown error does not invalidate coverage.
