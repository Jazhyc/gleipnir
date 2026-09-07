# Monitoring-only ID data scaling

Hypothesis: the standard monitor recipe may reach its best ID performance with
less than the full monitoring dataset. Train exactly 5%, 10%, 20%, and 50% of the
8,688 parents; reuse the completed 100% model. No auxiliary MIL/prefix/rationale
loss, deception data, new annotation, or extra learning-rate/epoch search.

User revised the initial two-epoch proposal to **one epoch for speed**, before
preparation or launch. Use the standard Qwen3.5-4B recipe: LR 2e-5, one epoch, seed 0,
rank128/alpha256 QLoRA, NF4 double quantization/BF16, FP32 master adapters,
AdamW, zero dropout/weight decay, linear decay/3% warmup, microbatch1/accum32,
linear-attention-only checkpointing and full-attention/linear-shell compilation.
Require pinned FLA0.5.2/causal-conv1d1.6.2.post1/Triton3.7.1 and a longest-context
preflight. Each run initializes fresh; no model is continued from another subset.

The exact rounded counts are 434, 869, 1,738, and 4,344, with 14, 28, 55, and
136 optimizer updates respectively. One epoch is fixed, **not GPU-time or update
count**. Data quantity and optimization exposure therefore change together.
The matched 100% reference has 272 updates and ID macro pAUROC@20 0.8710662298.
The two-epoch 0.8829517621 result is separate context, not a point on this curve.

Subsets are nested with seed-0 hash ordering within source/label strata and
incremental proportional-deficit allocation, preserving the original mixture
to rounding error. Do not reuse the older equal-capped source-balanced OOD
scaling subsets: their mixture and optimization schedules answer another question.
Every recorded lineage in the current artifact is a singleton (8,688/8,688).
Fail closed if that changes rather than split a shared lineage. This does not
claim that the available lineage identifiers capture every latent dependency.
Freeze selection IDs/checksums and source/label/token counts; retain original
parent and Kimi soft-target artifacts and check the existing ID separation.

Queue behind the **entire** fixed-compute prefix pipeline, including its ID
evaluation. Use the existing reserved Lambda GPUs only: 50% on GPU0 and the
20%, 10%, 5% jobs sequentially on GPU1, scheduled by expected work. A process-exit
dependency requires the predecessor's successful completion and no remaining
GPU compute processes; it never cancels unrelated jobs or launches new capacity.
This remote execution queue is not an agent heartbeat. No in-chat scheduling
tool is available; do not promise automatic agent follow-ups.

Evaluate final checkpoints only on the same 3,012-row ID suite with the same
compact binary-logit prompt. Run master/export serving parity before persistent
vLLM evaluation. Report all five macro and source pAUROC/AUROC, calibration,
threshold diagnostics and ties, plus sample/token counts, steps and training
runtime. No intermediate ID selection, OOD use, adaptive fractions, or automatic
promotion. The existing +0.005 pAUROC / no >0.01 source loss / no >0.005 Brier
regression rule is exploratory versus 100%; near-equal smaller models should be
reported as potentially economical, not proven equivalent without replication.

Fail closed on input/selection/model/recipe drift, numerical errors, missing
rows, failed kernel/long-context/serving gates or predecessor failure. Keep
negative results. Reuse the shared duration execution/summary machinery; one
experiment config and entrypoint supply subset-specific preparation and hooks.

## Launch record

Prepared and queued on `gleipnir-improvement` with one epoch for every new job.
Queue PID `185381` was verified sleeping on the process-exit dependency for
prefix pipeline PID `184536`; `queue_status.json` reported `waiting`. Only the
existing prefix trainer was using the two GPUs at verification. Selection
manifests and frozen campaign artifacts were pulled locally as a backup.
The focused scaling and shared-duration tests passed (17 tests).

From the remote repository root, preparation and execution use:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.monitoring_id_scaling.run prepare
PYTHONPATH=src .venv/bin/python -m experiments.monitoring_id_scaling.run queue
```

Do not repeat preparation or start a second queue for this campaign. Inspect
`results/monitoring_id_scaling/queue_status.json` and
`logs/lambda/monitoring_id_scaling/queue.log` for its current state.

## Two-GPU evaluation takeover

User authorized splitting the final adapter evaluations across both H100s.
Paused only queue supervisor 185381, allowed the last 50% trainer and export
to finish, verified no remaining GPU processes, then replaced the supervisor
with `python -m experiments.monitoring_id_scaling.resume_evaluation` (PID 191449).
No training was restarted. The original manifest is unchanged; the recorded
code-only transition extracts the identical curve-writing function for reuse.
All other manifested checksums and completed training metadata remain required.

Run the original master/export serving-parity gate, then two persistent TP1
vLLM engines: GPU0 evaluates 5% and 20%, GPU1 evaluates 10% and 50%. Adapter
ownership is disjoint, each evaluates all 3,012 rows, and existing predictions
resume in place. No data sharding/merging or frozen scoring-config change is
needed. Both workers must complete before the shared summary and scaling plot.
Fourteen focused assignment/scaling tests passed. Runtime assignments and code
provenance are in `evaluation_split.json`; logs are `evaluation_gpu{0,1}.log`
under the campaign root and `logs/lambda/monitoring_id_scaling/evaluation_split.log`.

## Completed results (2026-09-07)

All four adapters completed all 3,012 ID rows; both GPU lanes exited and GPUs
are idle. Predictions, summary and scaling_curve.json were collected locally.
Audit confirms 3,012 unique matching ID/source/label tuples per arm, finite
scores, frozen configuration hashes, and exact local metric recomputation.

| Data fraction | Rows | Macro pAUROC@20 | Macro AUROC | Macro Brier | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5% | 434 | .826454 | .949188 | .101404 | 449.84 |
| 10% | 869 | .855903 | .955862 | .083515 | 875.66 |
| 20% | 1738 | .886670 | .965245 | .075539 | 1742.11 |
| 50% | 4344 | .833644 | .941339 | .088836 | 4271.89 |
| Historical 100% | 8688 | .871066 | .957643 | .079273 | 9348.04 |

The 20% arm passes the exploratory gate: +.015604 macro pAUROC, with Gloom
+.019900 and STRIDE +.011307, and Brier improvement .003733. It takes about
29 minutes of training versus the historical 156 minutes. This is the best
observed one-epoch data fraction, not a proven optimal dataset size. Nested
subsets, a single seed and a fixed one-epoch schedule confound data content,
amount, number of optimizer updates and learning-rate evolution. The 50% dip
is concentrated in Gloom (pAUROC .696837 versus .803732 at 20%); STRIDE remains
strong (.970450 versus .969607). No automatic additional runs or OOD evaluation.

Threshold tradeoff: 20% macro recall at .5 increases from .847277 to .908383,
but macro FPR increases from .046872 to .110157; balanced accuracy is slightly
lower (.899113 versus .900203). Better ranking/Brier is not uniformly better
performance at the unchanged threshold. Relative to the separately evaluated
two-epoch full-data endpoint (.882952 pAUROC, .073997 Brier), 20% has slightly
higher ranking but worse Brier; these small differences are not replicated.
