# Frozen OOD reasoning audit

Question: do the 6,395 evaluated OOD inputs contain explicitly identified agent
thinking, or source reasoning fields accidentally serialized into the input?
Audit every frozen row without changing data, prompts, labels, model weights,
or evaluation results. This is an input audit, not a performance experiment.

Validate each author source parquet against the pinned hashes in
`prepare_teacher_ood_cache`, including its prompt/transcript equality. Match each
source transcript exactly to the frozen teacher and compact student prompts and
check actual Kimi K3 cache prompt hashes. Scan trajectories (not the monitor's
rubric or empty non-thinking chat wrapper) for thinking tools, tags, channels,
and headings. Inspect rendered assistant bodies and structured source content.

ControlArena exports have untyped assistant content in `trajectory_data`; check
whether it survives in the monitor transcript and distinguish submitted answers
from trivial substring collisions. CUA has flat transcripts; check whether all
assistant bodies contain only tool-call names. MLE has a separate `thinking`
column; compare complete traces, step bodies, and paragraphs of at least 80
characters, distinguishing overlaps inside submitted code fences. Long-paragraph
matching is a diagnostic, not a guarantee that every short fragment is absent.
Do not infer semantic absence of reasoning from absent tags alone.

Stop on hash, identity, or prompt mismatch. Record every marker or overlap for
manual review. `python -m experiments.ood_reasoning_audit.run` writes ignored
artifacts to `results/ood_reasoning_audit/`; conclusions belong in `docs/findings/`.
No inference, paid API calls, or remote experiment polling is needed.
