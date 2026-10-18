# SHADE-Arena length audit

## Question

Sinha et al. removed every input above 28,000 Qwen3-8B tokens. That cut 650 of
1,500 SHADE-Arena rows (43%) but at most 4% of any other strict-OOD source, so
the scored 850 rows are SHADE-Arena's shorter part. Does monitor quality fall
with length inside the retained rows? If so, the scored subset likely flatters
monitors relative to the full benchmark.

## Design

Post-hoc analysis of saved strict-OOD predictions; no inference and no
selection. Arms: trained Qwen3.8-27B seed 0, Kimi K3 (teacher prompt), and the
Gleipnir 4B teacher-seed SDPA seed 0. Length is the frozen workload's
`prompt_tokens` (Qwen tokenizer, binary student prompt) for every arm.

- Raw length quartiles per source, with bootstrap 95% intervals.
- SHADE-Arena only: quartiles by length rank within main task x attack
  generator (joined from upstream `extra_info` by `source_row_index`, with a
  label-alignment check), so every bin has the same task/generator mix.
- Per main task: within-generator length halves.

Group sizes are small (two tasks have 8 and 11 rows); treat per-task halves as
descriptive.

## Run

```bash
python -m experiments.shade_length_audit.analyze
pytest -q experiments/shade_length_audit
```

Writes `results/shade_length_audit/summary.json`. Interpretation lives in
[`docs/findings/shade_length_audit.md`](../../docs/findings/shade_length_audit.md).
