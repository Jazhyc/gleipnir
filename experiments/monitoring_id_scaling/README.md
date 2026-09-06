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
