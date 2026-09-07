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

## Completed result

Collected and validated 2026-09-08: all 3,012 rows completed on attempt zero,
with exact identity/hash coverage and successful master/serving parity.

| Source | Original pAUROC@20 | CoT removed | Prose and CoT removed |
| --- | ---: | ---: | ---: |
| STRIDE | 0.966052 | 0.928943 | 0.918835 |
| Gloom-Exfiltration | 0.772335 | 0.771224 | 0.687128 |
| Source macro | 0.869194 | 0.850083 | 0.802982 |

Macro AUROC changes from 0.957679 to 0.952674. STRIDE AUROC changes from
0.992866 to 0.983432; its Brier score worsens from 0.031929 to 0.056439.
At threshold 0.5, STRIDE recall falls from 0.956672 to 0.904679 and FPR rises
from 0.021680 to 0.032520. STRIDE has 44 classification flips (35 positive to
negative, nine negative to positive).

Unchanged Gloom provides a numerical-repeat control: mean absolute score change
0.003828, correlation 0.999677, six threshold flips (three each direction), and
pAUROC change -0.001112. This is consistent with inference/batching variation;
it is not an effect of CoT removal on Gloom. The considerably larger STRIDE drop
supports sensitivity to the removed reasoning. The combined-removal drop cannot
be attributed entirely to CoT. No retraining or threshold selection was performed.

Total input tokens fall from 34,631,573 to 33,750,959 (2.54%). Full ranking,
calibration, ties, paired diagnostics and artifact hashes are in
`results/id_cot_only_evaluation/comparison.json`. Cleaned prediction SHA-256:
`6cc4e931de18bbb40ec5846d8608077f536a0d3ccf09e5976ef4618f389f71e0`.
