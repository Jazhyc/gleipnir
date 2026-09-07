# Frozen OOD input reasoning audit

Status: completed 2026-09-08. Scope: all 6,395 frozen strict-OOD inputs, with
no changes to data, prompts, labels, evaluation results, or models.

## Finding

No confirmed private-thinking leakage was found in this OOD suite. In particular,
the STRIDE failure mode (source reasoning flattened into assistant prose) is not
present in the four ControlArena exports or CUA assistant sections. MLE-Sabotage
keeps its separate thinking traces outside the transcript while retaining
visible plans and submitted code. This supports a narrower conclusion than
claiming every natural-language fragment is semantically free of reasoning.

| Source | Rows | Evidence in actual monitor input |
| --- | ---: | --- |
| AgentDojo | 1,500 | All 6,380 assistant bodies empty; calls/results retained |
| IaC | 1,443 | All 27,734 assistant bodies empty; calls/results retained |
| Rogue Eval | 1,499 | All 3,533 assistant bodies empty; calls/results retained |
| SHADE-Arena | 850 | All 50,218 assistant bodies empty; calls/results retained |
| CUA-SHADE-Arena | 955 | All 24,571 assistant bodies contain only tool-call names |
| MLE-Sabotage | 148 | Plans/code retained; separate thinking fields excluded |

No explicit thinking tags, thinking channels, reasoning JSON keys, or named
thinking-tool calls were found in the transcripts. Two broad `Analysis:` heading
matches in CUA are tool-output task analysis (salary statistics and model-result
comparisons), not private assistant traces.

## Structured-source checks

The four ControlArena files retain 92,639 raw assistant messages in
`trajectory_data`, including 83,049 nonempty untyped assistant-content fields.
Some of that raw content clearly includes deliberation, but the rendered
transcripts omit all free-form assistant bodies. Full-content substring checks
find only 18 overlaps in `submit` arguments, manually reviewed as final task
reports, and three one-character collisions (`<` or `I`) in user/tool text.
There are no named `think` tool calls in these structured source exports.
The string-valued source content does not provide STRIDE-style typed boundaries.

All 148 MLE rows contain a separate nonempty `thinking` field. None of those
complete fields or their 225 complete step bodies occurs in the evaluated
transcript. Among 1,429 thinking paragraphs of at least 80 characters, 17 overlap
with the transcript, all in one row and all within submitted code fences.
Manual inspection confirms these are code fragments discussed during thinking
and also emitted as code, not prose thinking copied into the input. Visible
`Plan:` output remains part of the monitored behavior.

CUA provides flattened transcripts rather than original typed agent messages.
The `prompt` and `messages_with_policy` columns are monitor envelopes, not
additional raw agent messages. Every rendered assistant section contains only
`Tool calls:` and tool names. Tool outputs can contain task analysis or other
natural-language content; this audit does not classify all such content as
private or public reasoning based on semantics alone.

## Provenance and reproducibility

All six source files match their frozen hashes from
`adityaasinha28/deliberative-monitor-pipeline` at
`2f10f4e49f4bf349965640ac021406689455d1c3`, under `filtered_eval/`.
For every row, the source transcript equals the author's prompt content and
renders exactly to both the frozen teacher and student prompts. All 6,395
actual Kimi K3 cache entries match their teacher prompt hashes. This audits the
trajectory actually supplied for evaluation, not just unused metadata columns.

Teacher input SHA-256:
`fa4b0d917e51eb91a52efa93cddf0eb154b88993269c97cfdb161c42680136a9`.
Student input SHA-256:
`ee2baa18a5d36cda8f9d66df642b6f3c106f92ae8ce2813acca36fa9f0e4acd2`.

Run `python -m experiments.ood_reasoning_audit.run`. The
[experiment README](../../experiments/ood_reasoning_audit/README.md) defines the
scope and checks. Ignored `results/ood_reasoning_audit/summary.json` records
counts and source hashes; `rows.jsonl` records all marker/overlap evidence for
review. Three focused parser/location tests and Ruff pass.

This audit does not establish what bytes the paper's own model runs consumed,
or rule out unmarked reasoning fragments in arbitrary tool arguments/results.
It finds no evidence requiring a CoT-removal rerun of our OOD evaluations.
