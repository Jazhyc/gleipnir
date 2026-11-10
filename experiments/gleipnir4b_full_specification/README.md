# Gleipnir 4B scored with the full specification

User request, 2026-11-10 (technical report, appendix on instructions): the
report shows Gleipnir 27B scored with the compact instruction and with the
full specification (0.922 against 0.942 validation, 0.909 against 0.935
Mean-OOD, BF16). The only 4B counterpart is a pair of runs distilled from
Gleipnir 27B ([record](../gleipnir27b_teacher_4b/README.md)). The released,
Kimi-distilled 4B has been scored with the full specification only on an
adversarial set ([record](../adversarial_is_it_4b/README.md)).

- **Hypothesis.** The released Gleipnir 4B, trained with the compact
  instruction, does not gain validation or Mean-OOD pAUROC@20 when it reads
  the full specification, unlike the 27B.
- **Intervention.** Score the released 4B (`Jazhyc/Gleipnir-4B-BF16`, the QAD
  adapter) on the full-specification prompts that the 27B is scored on:
  `data/optimized_gap/teacher_id_workload.jsonl` (3,012 rows) and
  `data/optimized_gap/teacher_ood_workload.jsonl` (6,395 rows). The two
  models share the tokenizer and chat template, and `score.py` stops if a
  served token count differs from the workload's. Two arms:
  - `bf16`: stock vLLM 0.31 BF16 (`release_serving/stock_vllm.py` recipe);
  - `optimized`: the released optimized server (`release_serving/serve.py`).
- **Baselines.** The recorded compact-instruction rows of the same checkpoint
  on the same row IDs: BF16 `results/qad-v02-bf16-01/evaluation/` (0.9332
  validation, 0.8039 Mean-OOD) and optimized
  `results/qad-v02-optimized01/evaluation/` (0.9117, 0.7703). Not rerun.
- **Selection.** None. One checkpoint, no training, nothing is promoted and
  the released instruction does not change on this result. The OOD suite is
  reported once per arm.
- **Caveats.** The QAD step matched this adapter to its BF16 scores under the
  compact instruction, and the optimized server's compile caches were traced
  on compact-instruction workloads. The optimized arm therefore mixes the
  instruction change with any serving change on longer prompts.
- **Stop** on a token-count mismatch, a non-finite score, incomplete
  coverage, a server that fails its startup checks, or completion.
- **Compute.** One B200 Pod from `ghcr.io/jazhyc/gleipnir-serve:0.2`, about
  one hour. Not launched: needs the user's instruction.

```bash
# On the Pod, in /workspace/gleipnir, with one server running on port 8000:
PYTHONPATH=src:. $PY -m experiments.gleipnir4b_full_specification.score \
  --backend {bf16|optimized} --output results/gleipnir4b_full_specification/<arm>
# Anywhere with the result rows:
PYTHONPATH=src:. python -m experiments.gleipnir4b_full_specification.compare
```

## Result (2026-11-10, US-NC-2 B200 `GPU-480a554d…`, Pod `xaqw5n348zwok3`)

Only the `bf16` arm ran. The user dropped the `optimized` arm before launch:
the report compares instructions in BF16 only, to isolate the instruction.

Stock vLLM 0.31 from the release image (`ghcr.io/jazhyc/gleipnir-serve:0.2`,
`python -m vllm.entrypoints.cli.main`; the image's `vllm` script has a stale
shebang). Parity on this host first: the compact validation rows score 0.9333
against the recorded 0.9332 (MAE 0.0038, correlation 0.99989, 106K prompt
tokens/s). Inputs matched the stored SHA-256.

| Released 4B, BF16 | Validation pAUROC@20 | Mean-OOD pAUROC@20 |
|---|---:|---:|
| Compact instruction (recorded) | 0.9332 | 0.8039 |
| Full specification | 0.9247 | 0.8032 |
| Difference | −0.0084 | −0.0007 |

| Source | Compact | Full specification |
|---|---:|---:|
| Gloom exfiltration (validation) | 0.8945 | 0.8781 |
| Test STRIDE (validation) | 0.9719 | 0.9714 |
| AgentDojo | 0.9462 | 0.9314 |
| CUA-SHADE-Arena | 0.7418 | 0.7460 |
| IaC | 0.8618 | 0.8930 |
| MLE-Sabotage | 1.0000 | 1.0000 |
| Rogue Eval | 0.9742 | 0.9723 |
| SHADE-Arena | 0.2993 | 0.2764 |

- The 4B does not gain from the full specification: −0.8 points on
  validation and −0.1 on OOD, against +2.0 and +2.6 for the 27B.
- Scores move without a ranking gain: mean absolute score difference 0.029
  (validation) and 0.040 (OOD), 77 and 285 rows crossing 0.5. IaC gains 3.1
  points and AgentDojo and SHADE-Arena lose 1.5 and 2.3.
- One checkpoint and one pass per instruction; no intervals.
- Rows: `results/gleipnir4b_full_specification/bf16/`,
  `comparison.json`; driver and logs: `logs/runpod/report_ablations/`.
