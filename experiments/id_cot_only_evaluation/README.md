# ID evaluation with only identified CoT removed

Hypothesis: the standard Gleipnir 4B checkpoint's ID performance changes when
source-typed agent reasoning and explicit `think` calls are removed while visible
assistant prose remains. Compare all 3,012 fixed ID rows with the original inputs
and the completed combined prose/thinking removal. No training, OOD evaluation,
checkpoint selection, or threshold tuning. Stop on source alignment ambiguity,
remaining identified reasoning, input drift, parity failure, or evaluation failure.

The transformation deletes exact reasoning spans from assistant bodies and
source-validated named `think` calls. Other bytes, including prose, tool results,
and non-thinking calls/arguments, are preserved. Every rendered assistant body is matched against its structured blocks;
a visible text copy identical to reasoning is retained. Empty assistant headers remain to avoid other
format edits. Gloom has no identified thinking and remains byte-for-byte unchanged;
this does not establish absence of unmarked reasoning. Raw STRIDE is checksum-pinned
by the existing audit's source reader. Per-row removal counts are saved locally.

Reuse the prior frozen checkpoint, prompt, model revision, FlashInfer engine,
threshold 0.5, and 128-row batch partition across both reserved Lambda H100s.
The shared runner records idle hardware and performs fresh master/serving parity
before evaluating. All config, prompt, manifest, and adapter identities are checked.
Report source-macro/per-source and pooled ranking, Brier, threshold diagnostics,
and score ties. Historical comparisons may include BF16/batching variation;
unchanged Gloom is a useful control for that variation.

Run `python -m experiments.id_cot_only_evaluation.prepare`, sync the new experiment,
modified shared helpers, inputs and prepared result root using `scripts/lambda_cloud.py`,
then launch `bash experiments/id_cot_only_evaluation/launch.sh` on improvement.
After collecting results, run `python -m experiments.id_cot_only_evaluation.summarize`.
Artifacts: `results/id_cot_only_evaluation/`; logs: `logs/lambda/id_cot_only_evaluation/`.
No scheduling tool is available; agent monitoring uses active-turn checks.
