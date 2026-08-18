# AISI lie-detection external transfer

This experiment evaluates whether the selected Gleipnir monitors transfer to
the main testbeds from *“Did you lie?” Evaluating Lie Detectors across Model
Scale and Belief-Verified Model Organisms* (`arXiv:2606.12618v2`). It is a frozen
external evaluation, not another hyperparameter-selection surface.

## Hypothesis and models

The hypothesis is that the mixed soft-plus-hard monitor transfers better than
either supervision source alone, especially on belief-verified trained model
organisms. Score the unadapted `Qwen/Qwen3.5-9B` base once and all three frozen
seeds of:

- rank-128 QLoRA soft-only training (`strength_id=soft-only`);
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
stop condition is ten complete vLLM evaluations with matching input hashes: one
base plus three seeds each for soft-only, hard-only, and mixed training. The
soft-only baseline is part of the intended supervision ablation; its initially
omitted external evaluation is a campaign-completeness correction, not a new
result-conditioned intervention.

## Backend parity and execution

Frozen text-only evaluation uses vLLM continuous batching with one constrained
output token and explicitly requested logprobs for the literal `0|1` tokens. Do
not reuse the training microbatch for the full evaluation. Transformers eager
inference is limited to a 64-row parity canary: compare the causal FP32 master
adapter with its checksum-rebased vLLM serving copy, require Pearson score
correlation of at least 0.99, mean absolute score difference at most 0.02, and a
nonzero adapter effect under both backends. Small numerical differences are
acceptable; an architecture/key-layout mismatch is not.

The launcher prepares derived JSONL data from the pinned Parquet files, verifies
FLA for the eager parity half, requires the parity gate to pass, and then
evaluates two continuously batched vLLM lanes on the reserved Lambda H100s:

```bash
python experiments/aisi_lie_detection_transfer/run_lambda.py
```

The full result metadata records vLLM version, requested token IDs, memory
utilization, adapter source/destination checksums, and measured throughput.
Derived inputs, manifests, predictions, summaries, and logs remain in ignored
`data/`, `results/`, and `logs/` trees. No upstream dataset copy or model weight
is committed.

## Outcome

The ten-run evaluation completed on 2026-08-18. The three-seed soft-only model
reached `0.95283` pooled AUROC, `0.86452` pooled balanced accuracy, `0.85347`
subject-macro AUROC, and `0.76640` subject-macro balanced accuracy. It exceeded
the mixed weight-0.50 model on pooled AUROC (`0.94795`) and Varied Deception
macro AUROC (`0.96057` versus `0.95823`), despite the mixed objective's clear
advantage on the repeatedly inspected internal development set. The external
result therefore does not support treating the hard anchor as a broad transfer
improvement.

Neither adapted objective transfers uniformly. Soft-only falls from the base
model's `0.82011` Gender Secret AUROC to `0.69057`; mixed falls further to
`0.66615`. Mixed improves over soft-only on AuditBench and Sandbagging and is
nearly tied on the 45-subject macro (`0.85334` versus `0.85347`). Routing the
four Gender Secret subjects to the base gives soft-only `0.86498` subject-macro
AUROC and mixed `0.86702`. Treat this source-aware routing result as a diagnostic
of unidentifiable transcripts, not a general deployment solution.

Two evaluations stalled nondeterministically inside vLLM generation at
different prompt percentages. In each case the process stayed live, GPU
utilization reported 100% at unusually low power, logs stopped advancing, and
there were no NVIDIA XIDs, ECC errors, or thermal faults. A fresh process, GPU,
and compilation-cache directory completed the identical seed and passed the
previous stall point. This implicates the CUDA-graph/Qwen3.5 hybrid
linear-attention runtime path rather than the dataset, adapter, SSH transport,
or GPU hardware. Discard partial stalled runs, isolate adapters in fresh
processes and caches, and add a progress watchdog; use eager execution as the
next diagnostic if the stall recurs.
