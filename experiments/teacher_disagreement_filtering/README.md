# Teacher-disagreement filtering sweep

Hypothesis: removing confident Kimi/source-label conflicts improves ID monitoring
ranking while preserving ambiguous soft supervision. Disagreement is not proof of
teacher or source-label error; authorization/rubric disputes remain possible.

Train Qwen3.5-4B on original monitoring data only: no added injection views,
deception mixture, hard-label loss, target rewriting or deduplication. The five
arms are the reused unfiltered SDPA control and removal deltas 0, 0.05, 0.1 and 0.2.
For delta zero, remove when `int(p_harmful >= 0.5) != label`. For positive delta,
remove label-1 rows below `0.5-delta` and label-0 rows above `0.5+delta`; preserve
band-edge equality. Filter student and teacher rows together by dataset/index,
preserving retained bytes and order. Record exact-content groups split by the
rule; do not silently expand exclusions beyond the requested probability rule.
No derived views exist in this campaign. Any future holdout must group duplicate
content and task/conversation lineage.

Expected removal counts are 852, 796, 723 and 627 of 8,688 original records.
Freeze counts, source/class composition, tokens, duplicate-group diagnostics,
input hashes and all arm configurations before training. Cached full-census
token lengths must match original identity, trajectory and student prompt hashes.

Use seed 0, the preserved zero-B rank-128/alpha-256 FP32 initialization,
BF16/SDPA/FlashQLA, AdamW 5e-5, one epoch, linear decay/3% warmup, batch 32,
16,384 packing budget and 29,696 context cap. Reuse the validated startup receipt
and persistent compiler caches. This user-requested precision/attention choice
is an exception to the FP4/FA4 default. Filtering changes updates, tokens and
schedule length; the sweep does not isolate selection from reduced exposure or
source-mixture changes. No matched random-removal arms are included in this
initial user-selected sweep. One seed does not measure seed variance.

Select only each completed final one-epoch checkpoint. Score all 3,012 canonical
CoT-removed ID rows with BF16 vLLM 0.31, causal LAST two-logit scoring, existing
128-row grouping and c128. BF16 serving is an explicit experimental exception
to the optimized default. Require fresh adapter-specific FP32-master/BF16-merge/
BF16-serving agreement and native BF16 audits. Preserve FP32 adapters on the
network volume. Reuse one `/tmp/gleipnir-merged/teacher-disagreement-sweep` slot:
retire the completed scorer, verify weights and persistent masters, preserve its
merge manifest/receipts, then remove the disposable merged files before the next
arm. Keep only the final merge. Superseded merge-stage receipts retain their
original checksums; those files require reconstruction rather than direct resume.

Reuse the completed `student_injection_awareness/4b/regular` control checkpoint,
bound by master/export/config/metadata/job hashes. Its initial tensor fingerprint,
data/targets, seed, objective, optimizer and SDPA recipe match; it passes the
current completion validator. Retain its original metadata and explicitly record
`training_performed_this_run=false`. Fresh master/merge/serving gates and ID scoring
match the filtering arms' serving setup. Historical source/runtime/host and
unmeasured training-repeat variation remain qualifications. This replaces the
redundant fresh control requested to be reconsidered by the user.

Primary comparison: source-macro raw normalized pAUROC@20 against this control.
Report source/pooled AUROC, calibration, threshold diagnostics,
ties and training exposure. No early stopping, checkpoint selection or automatic
promotion. Report every arm, including negative results. Strict OOD and APPS are
outside this ID-only campaign; an ID winner does not establish transfer.

`prepare --gpu-uuid UUID` writes immutable inputs/configs under
`data/teacher-disagreement-{control,delta000,delta005,delta010,delta020}-sdpa04/`
and a sweep manifest in `results/teacher_disagreement_filtering/sweep04/`.
`run` inventories and prepares all five campaigns before running them sequentially
through the shared monitoring campaign runner. It saves queue state, logs and an
aggregate summary automatically. Each arm has a three-hour wall-time limit;
failure stops the queue and preserves receipts, with no automatic retraining or
parity waiver. Only owned subprocesses are retired on timeout. A completed
scorer is retired before the next training arm and after the final evaluation.
The Pod stays running after completion/failure; no automatic lifecycle mutation.
The replaced `sweep01` queue and its partial preparation/startup receipts are
preserved; its owned processes were retired before replacing the control.
`sweep03` stops before launching an arm because the retired queue's control log
already exists. The replacement isolates queue logs by sweep attempt and records
outer preparation/launch failures explicitly. Preserve both failed/replaced receipts.

The user explicitly requests unattended execution after initial checks, overriding
recurring agent follow-ups for this campaign. This session has no agent scheduler.
The detached runner is process automation, not promised future agent monitoring.

```bash
PYTHONPATH=src:. python -m experiments.teacher_disagreement_filtering.prepare \
  --gpu-uuid UUID
PYTHONPATH=src:. python -m experiments.teacher_disagreement_filtering.run
```
