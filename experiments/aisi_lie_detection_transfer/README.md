# AISI lie-detection external transfer

This experiment evaluates whether the selected Gleipnir monitors transfer to
the main testbeds from *“Did you lie?” Evaluating Lie Detectors across Model
Scale and Belief-Verified Model Organisms* (`arXiv:2606.12618v2`). It is a frozen
external evaluation, not another hyperparameter-selection surface.

## Hypothesis and models

The hypothesis is that the mixed soft-plus-hard monitor transfers better than
hard-only training, especially on belief-verified trained model organisms. Score
the unadapted `Qwen/Qwen3.5-9B` base once and all three frozen seeds of:

- rank-128 QLoRA hard-only training (`strength_id=hard-only`); and
- the internally selected soft-loss 1.0 / hard-loss 0.50 mixture
  (`strength_id=0500`).

Do not tune thresholds, prompts, seeds, adapters, or dataset inclusion after
viewing these results. The literal `0|1` boundary and threshold 0.5 remain fixed.

## Dataset contract

Source `ai-safety-institute/lie-detection-rollouts` is pinned to revision
`177a81666ab378ce9d055507bc30f480ab63e74b`. The Hugging Face collection is
`ai-safety-institute/lie-detection-69e8e7b608f926010851f645`. The repository
declares license `other` and contains no separate license/terms file; record that
qualification with every materialization.

Use the full official `varied_deception` evaluation split for all 32 base subject
models in the paper's per-model Table 9. Exclude `varied_deception_validation`,
which the paper identifies as the disjoint detector-tuning pool. Also use the
full released auditing splits for the 13 paper model organisms:

- AuditBench: three Qwen3.5 and four Qwen3.6 organisms;
- Gender Secret: female and male organisms on each backbone; and
- Sandbagging: one organism on each backbone.

These auditing splits are small testbeds explicitly constructed for detector
evaluation, not unsplit training corpora, so a new 10% sample would only add
variance. If a future source offers only a combined train corpus with no held-out
partition, add a deterministic, lineage-grouped 10% test partition before using
it here.

Map `honest` to 0 and `lie` to 1. Exclude `ambiguous` and null labels. Remove all
system messages, matching the paper's judge-model condition so prompted-lying
instructions cannot reveal the label. Exclude `reasoning_content` because this
is a text-output monitor, not a chain-of-thought judge. Materialize prompts with
the unchanged Gleipnir monitor template in `prompt.txt` and record its hash.

Before evaluation, compare visible transcript and final-user hashes with the
student training corpus. Remove exact visible-transcript matches. Retain
question-only overlaps with an explicit `seen_internal_user` flag so results can
be diagnosed rather than silently redefining the paper population.

## Metrics and aggregation

Report AUROC, balanced accuracy, Brier score, score ties, FPR, and recall for
every subject-model/testbed group. Report macro means within each of Varied
Deception, AuditBench, Gender Secret, and Sandbagging; a macro mean over all
subject/testbed groups; pooled metrics; and seen-versus-unseen-question
diagnostics. Aggregate adapter metrics over the three fixed seeds.

The overall subject macro is dominated by the 32 Varied Deception subject
models, so testbed-level metrics are the primary transfer interpretation. The
stop condition is seven complete causal evaluations with matching input hashes:
one base, three hard-only seeds, and three mixed seeds.

## Execution

The launcher prepares derived JSONL data from the pinned Parquet files, verifies
the pinned FLA kernels, and evaluates two lanes on the reserved Lambda H100s:

```bash
python experiments/aisi_lie_detection_transfer/run_lambda.py
```

Derived inputs, manifests, predictions, summaries, and logs remain in ignored
`data/`, `results/`, and `logs/` trees. No upstream dataset copy or model weight
is committed.
