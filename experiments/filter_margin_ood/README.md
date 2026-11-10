# OOD scores for the teacher-agreement filter margins

User request, 2026-11-10 (technical report, appendix on the filter): the
[margin sweep](../teacher_disagreement_filtering/README.md) scored its five
4B arms on the validation (ID) set only. The
[OOD follow-up](../teacher_disagreement_ood/README.md) scored only the strict
filter (0.7869) and compared it with a cached unfiltered score (0.7762) from a
different serving path. This experiment scores all five arms on the OOD suite
through one serving path.

- **Hypothesis.** Descriptive: the report states that the filter margin was
  chosen on the validation set, and this gives the OOD score of each arm. No
  direction is predicted for the margins.
- **Intervention.** None in training. For each arm (unfiltered control,
  margins 0, 0.05, 0.10, 0.20; `teacher-disagreement-*-sdpa04`), rebuild the
  merged BF16 checkpoint from the pinned base and the arm's serving adapter
  with `gleipnir.adapters.merge.merge_checkpoint`, and require every output
  file to match the SHA-256 in the arm's recorded `merged_artifact.json`. The
  arm's recorded master-to-merged parity then applies.
- **Scoring.** Stock vLLM 0.31 BF16 (`release_serving/stock_vllm.py` recipe),
  compact instruction, batches of 128 rows: first the 3,012 validation rows,
  then the 6,395 OOD rows (`data/teacher-disagreement-strict-ood01/workload.jsonl`).
- **Gates.** Before an arm's OOD rows are scored, its validation scores must
  agree with the arm's recorded rows: mean absolute difference at most 0.01
  and macro pAUROC@20 within 0.005. The margin-0 arm's OOD scores are also
  compared with its recorded OOD rows (0.7869). The released 4B passed the
  same stock recipe with 0.004 and 0.0003.
- **Selection.** None. The margin stays the one chosen on the validation set
  (margin 0); the OOD scores are reported once and select nothing.
- **Stop** on adapter checksum drift, a merge that differs from the record, a
  failed gate, a token-count mismatch, a non-finite score or completion.
- **Compute.** The same B200 Pod as
  [gleipnir4b_full_specification](../gleipnir4b_full_specification/README.md),
  authorized by the user on 2026-11-10. The adapters are on the US-NC-2
  volume `ixbh81vf9c`.

```bash
# On the Pod, per arm:
PYTHONPATH=src:. $PY -m experiments.filter_margin_ood.run --arm delta005 \
  --adapter <volume>/results/teacher-disagreement-delta005-sdpa04/4b/monitor/model \
  --base <Qwen3.5-4B snapshot>
# Anywhere with the result rows:
PYTHONPATH=src:. python -m experiments.filter_margin_ood.compare
```

## Result

Not run yet.
