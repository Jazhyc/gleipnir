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

Run 2026-11-10 on B200 Pod `xaqw5n348zwok3` (stock BF16 vLLM 0.31; deleted
after the artifacts were pulled and checksum-verified). Every merge matched
its recorded artifact and every arm passed the validation gate (score MAE
0.0041 to 0.0049, pAUROC@20 within 0.0005 of the record). Source-macro
pAUROC@20 (`results/filter_margin_ood/comparison.json`):

| Arm | Removed | Validation (3,012) | OOD (6,395) |
|---|---:|---:|---:|
| control (unfiltered) | 0 | 0.8463 | 0.7757 |
| delta000 (strict, selected) | 852 | 0.8963 | 0.7875 |
| delta005 | 796 | 0.8788 | 0.7956 |
| delta010 | 723 | 0.8366 | 0.7672 |
| delta020 | 627 | 0.8509 | 0.7785 |

- delta000 reproduces its recorded OOD score (0.7869; score MAE 0.0042).
- The control's OOD score equals the cached reference to three decimals
  (0.776), so the serving-path caveat on that number no longer applies.
- delta005 scores 0.008 above the strict filter on OOD while scoring 0.017
  below it on validation. The margin was selected on validation before this
  run, and this result does not change the selection.
- Neither split is monotone in the margin. The OOD spread across arms
  (0.767 to 0.796) is within the three-seed range of the strict recipe
  (0.777 to 0.813), so one seed per arm cannot rank the margins on OOD.
