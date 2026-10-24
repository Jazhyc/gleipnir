# Linear decoding of backdoor status under identical text

Protocol frozen before extraction. Implementation only, no GPU or paid API run
is authorized by this change. Step 1 results at `76a112c` motivated this study,
so this is a follow-up with prior outcome exposure, not a new untouched test.

## Question and interpretation

Does the v0.3 4B monitor retain linearly accessible information about which
program is backdoored when identical accusations collapse its output AUROC?
A successful held-out decoder establishes accessible information in the sampled
residuals. It does not establish that the monitor uses that information, or
that the probe detects semantics rather than correlated code features. A failed
probe does not establish erasure: the signal may be nonlinear, elsewhere, or
unrecoverable with 50 training tasks. APPS remains excluded from monitor weight
training and checkpoint selection. This diagnostic fits only an offline decoder,
an explicit task-authorized exception to the general APPS fitting restriction.

## Frozen data and splits

Read only `results/adversarial_crossed_text_amendment3_20261023/variants.jsonl`
and `selection.json`, pinned by SHA-256 in `config.json`. Keep original task IDs,
labels and exclusions. No new generation, execution of submitted code, or
replacement tasks. Step 1 has 50 development and 99 usable held-out tasks;
selection lists 100 held-out IDs, with invalid-trigger task 1099 absent.

Extract all 100 development no-text rows. Use five task-grouped folds: sort
50 development task IDs by SHA-256 of `probe-fold-20261023:` plus task ID, then
assign by sorted index modulo five. Both programs always stay together. Fit
and choose layer, pooling and regularization using only these no-text folds.
Refit once on all 50 development tasks. Save the selection and fitted parameters
before held-out extraction. Never tune on development text conditions or on
held-out labels. The canary may inspect development text solely for numerical
and token-span checks, never for decoder selection.

Test that fixed decoder on held-out no text, irrelevant text, specific
accusation and generic accusation: 99, 93, 81 and 99 pairs respectively.
Cover story has 28 pairs and is descriptive only. Omit task restatement and
calibration rows from probe extraction and fitting. This is 900 rows per arm,
including 100 development and 800 held-out rows. Report every missing row and
retain only complete honest/backdoored pairs for metrics. Primary reports
require a complete extraction, with no silently dropped computational failures.
The text-validity exclusions are inherited and are not random. Uncertainty is
conditional on this text/donor collection and the fitted decoder, excluding
writer, training-seed and shared-donor variation. The held-out tasks were already
used by Step 1 and may have earlier no-comment scores.

## Models, capture and canaries

Use `gleipnir.adversarial.scorer.Scorer` directly. Required arms are
`4b-v03_student` and `4b-base_student`, both with Step 1's exact student prompt.
Optional `27b-merged_teacher` uses its teacher prompt. The model revision pins
come from `gleipnir.adversarial.monitors` and must match Step 1 runtime receipts.
Reuse Step 1 BF16 Transformers/PEFT/SDPA with FLA and fla-core 0.5.2, one model
resident, one unpadded prompt per forward, `logits_to_keep=1`, no gradients.
Do not merge the adapter, substitute vLLM, change cache behavior, or truncate.
Package versions must match the archived Step 1 runtime. No API fallback.

Capture decoder-block output residuals after blocks 8, 16, 24 and 32 in the
32-block 4B; optional 27B uses blocks 16, 32, 48 and 64 in its 64-block model.
These are one-based completed blocks, hooks on indices 7/15/23/31 or
15/31/47/63, before the final model RMSNorm. Fail on architecture mismatch.
At each block save the final prompt token residual, the decision position just
before predicting `0` or `1`, and the mean over submitted executable-code
region tokens. The latter excludes the inserted module docstring and framing,
but includes original comments and whitespace. Token offsets must come from
the exact full rendered prompt, agree with `Scorer.encode`, and lie wholly in
the code span. Exclude boundary-straddling tokens. If offsets are unavailable
or code cannot be aligned, stop; any decision-only amendment must be frozen
before extraction, never chosen from performance. Pool in FP32 and save FP32.
Hooks retain only pooled vectors, never all token-by-layer residuals.

Before every extraction invocation, replay the original Step 1 eight-row
canary, including v0.3 score drift at most 0.02 and nonzero adapter effect.
Check finite raw logits as well as probabilities. Then compare hooked and
unhooked scoring on both programs of the first development no-text pair and
first available development specific-accusation pair: absolute score drift at
most 1e-6, prompt/token identity exact, finite captures, all eight vectors
present. Save successful and failed receipts. Check the campaign deadline
before loading, after loading, and before and after every forward, including
canaries. Stop on failed parity, nonfinite data, identity change, timeout or OOM.

## Decoder and controls

Standardize each activation coordinate using fitting-fold statistics only.
Use L2 logistic regression with intercept, liblinear, tolerance 1e-6,
maximum 5000 iterations, seed 0, and C in `[0.01, 0.1, 1.0]`. Choose the tuple
with highest mean five-fold no-text validation AUROC. Ties use earlier layer,
decision position before code mean, then smaller C. Refit the selected tuple
on all development no-text rows. Report all CV candidates and freeze the fitted
parameters and their input identities. Abort on convergence warnings. No PCA,
feature selection, per-condition refit, sign flipping or held-out selection.

Controls use the same task folds and no-text training:

- One shuffled-label probe. Independently swap honest/backdoored labels within
  each development task by a fixed seeded coin, preserving pair balance. Run
  the same full layer/pooling/C selection against those shuffled labels, then
  test against true held-out labels. Save the permutation. This is a negative
  control, not a permutation-test p-value or a guarantee of AUROC exactly 0.5.
- A code-length logistic baseline with log1p UTF-8 byte length and line count
  of the original no-text program, carried unchanged across its text views.
  Select C using the same folds. It tests simple length shortcuts, not all
  possible syntactic confounds.
- The identical probe pipeline on the untrained base, independently selected
  on the same development tasks. Report selected settings for both models.

For each held-out condition, report pooled AUROC, within-task win rate with
half credit for ties, and 95% percentile task-bootstrap intervals from 10,000
resamples with seed 0. Resample entire pairs, preserving multiplicities.
Compare probe AUROC to its own model's normalized binary output AUROC from
the same hooked forward on exactly the same rows. Also report raw-logit-margin
AUROC, since sigmoid saturation can create ties. Bootstrap paired AUROC
differences, including probe minus output and probe minus length. Compare base
and tuned probe/output on common task-condition rows with paired intervals,
including the between-model difference in probe-minus-output gaps.
Use logistic decision scores for probe ranking to avoid sigmoid saturation.
Save row predictions and fitted coefficients. These are descriptive estimates
without multiple-comparison correction; no checkpoint or prompt is promoted.

## Frozen qualitative predictions

1. The tuned 4B's selected no-text decoder will retain above-chance held-out
   discrimination under both accusation conditions and outperform its own
   output there. This would support accessible information beyond the readout,
   not causal use or a guarantee that all relevant information survived.
2. The base will retain accusation-condition decoding as well. Its probe minus
   output gap will be smaller than the tuned model's gap.
3. True-label probes will outperform the shuffled probe and length baseline
   under accusations. Length may still carry signal; a weak difference would
   qualify the semantic interpretation.
4. Optional 27B decoding and output will both separate accused programs, with
   a smaller probe-output gap than tuned 4B. Cover stories have no directional
   prediction and cannot determine any selection or claim.

Report misses and uncertainty. These predictions were informed by published
Step 1 output results, not by activations from this study.

## Separate optional causal follow-up, not implemented or authorized

After this observational study, freeze a new protocol on separate development
interventions. Candidate: patch the selected decision residual or its learned
one-dimensional projection between matched honest/backdoored programs under
identical text, then measure changes in the original output margin. Include
unchanged/self patches, equal-norm random directions, reversed patches,
no-text controls and off-direction patches. Choose strength and layer only on
development tasks and evaluate a frozen choice on new held-out tasks. Alignment
of sequence positions, off-manifold shifts and collateral effects need explicit
controls. Decoder success alone must not be described as evidence of use.

## Bounds, storage and commands

`extract.py` is disabled unless passed `--execute`, requires a clean committed
protocol, and never starts, stops or contacts a VM. The mandatory campaign file
is `results/adversarial_campaign_20261023/vm.json`. Its current deadline is
2026-10-23 15:00:27 UTC and has expired. Do not edit it to bypass the guard.
A later run requires an authorized campaign window and matching external
shutdown timer managed by the campaign owner. Each extraction invocation also
has a 3600-second wall limit including loading and canaries. The shared campaign
deadline always wins; no resume extends it. Run only after separate GPU approval.

GPU estimate, not a benchmark: the archived Step 1 student-token counts give
858,344 tokens for the 900-row grid, mean 954 and maximum 2,923. Teacher
prompt lengths differ slightly. Budget 10–25 minutes per 4B and 20–45 minutes for optional
27B, including load/canaries; two required models about 20–50 minutes total.
Step 1 completed five arms in roughly 38 minutes on A100 80GB, but Python hooks
and pooling can slow this run. Log actual prompt tokens/s and peak allocated
GPU bytes. BF16 base weights are about 8 GB for 4B, plus the large FP32 PEFT
adapter and temporaries; budget 16–30 GB for tuned 4B, 12–20 GB for base and
60–75 GB for 27B. Long rows and kernels can exceed these estimates; stop on OOM.
No full hidden-state stack is requested. Host RAM should be at least 16 GB.

FP32 captures occupy `rows * 4 layers * 2 pools * hidden width * 4 bytes`:
73,728,000 bytes per 4B, about 70.3 MiB, at width 2560; optional 27B at width
5120 needs about 140.6 MiB. Both required arms total 140.6 MiB; all three about
281.3 MiB. Per-row NPZ containers, metadata, model coefficients and checksums
add overhead; reserve 1 GB disk excluding model/cache/data files. Preserve
persistent model/compiler caches. No weights or input programs enter Git.

From the upstream repo root, after approval, using the existing Step 1 VM
Python environment, with no environment rebuild or `.env` access:

```bash
export PYTHONPATH=src:.
python -m experiments.adversarial_crossed_text_probe.extract --arm 4b-v03_student --split development --execute
python -m experiments.adversarial_crossed_text_probe.analyze fit --arm 4b-v03_student
python -m experiments.adversarial_crossed_text_probe.extract --arm 4b-v03_student --split heldout --execute
python -m experiments.adversarial_crossed_text_probe.analyze evaluate --arm 4b-v03_student
python -m experiments.adversarial_crossed_text_probe.extract --arm 4b-base_student --split development --execute
python -m experiments.adversarial_crossed_text_probe.analyze fit --arm 4b-base_student
python -m experiments.adversarial_crossed_text_probe.extract --arm 4b-base_student --split heldout --execute
python -m experiments.adversarial_crossed_text_probe.analyze evaluate --arm 4b-base_student
python -m experiments.adversarial_crossed_text_probe.analyze compare --arms 4b-v03_student 4b-base_student
```

Optional 27B uses the same four arm commands with `27b-merged_teacher`, only
if separately chosen before its extraction and within the campaign deadline.
Offline fitting/evaluation need NumPy and scikit-learn, never torch or a GPU.
Repeated extraction resumes verified completed rows; every invocation repeats
canaries. Incomplete rows do not count as complete. Failures stay in receipts.

Sync only `experiments/adversarial_crossed_text_probe/` as new source, or
transfer its path-scoped commit as a Git format-patch and apply with `git am`.
The new files must be committed and clean in the VM checkout; plain rsync of
untracked files will fail the execution gate. Existing VM dependencies must
already match the pinned Step 1 source hashes. Required
read-only inputs: Step 1 `variants.jsonl`, `selection.json`,
`variants.manifest.json`, `scores/*runtime.json`; the original eight-row canary
reference `results/adversarial_phase1_v03_replay/queries.jsonl`; original
`data/apps/prompts/eval_neutral.jsonl`; both instruction files under
`experiments/tool_trajectory_monitoring/prompts/`; and the campaign-owned
`vm.json`. Shared code imports from `src/gleipnir/adversarial/`,
`src/gleipnir/evaluation/`, `experiments/adversarial_code_or_comments/` and
`experiments/adversarial_crossed_text/` are read-only. Never overwrite Step 1
source, another experiment, campaign metadata, model caches, or `.env` during
sync. Pull back all of ignored `results/adversarial_crossed_text_probe/`,
including NPZ files, row journals, manifests, fits, canaries and reports.

Local focused checks, CPU only:

```bash
PYTHONPATH=src:. python -m pytest -q experiments/adversarial_crossed_text_probe/test_probe.py
ruff check experiments/adversarial_crossed_text_probe
```

## Implementation validation

CPU checks on 24 October 2026: 27 synthetic tests passed; Ruff passed. Tests
cover the failed-canary gate, hooked/unhooked canary drift, deadline checks,
block indexing and hook cleanup, token boundaries and docstring exclusion,
task folds, development-only selection, shuffled pair labels, portable fitting,
bootstrap multiplicities and ties, full artifact analysis, fit locking and
corruption rejection. The real pinned input audit checked all 900 code spans
against their no-text originals and verified Step 1 source hashes. No model,
GPU or VM was accessed, and no activations or scientific results were produced.
Actual Qwen hooks, tokenizer offsets, GPU memory and timings remain untested.

The isolated local test environment used Python 3.12, NumPy 2.5.4,
scikit-learn 1.9.1, pytest 9.1.1 and Ruff 0.17.2. This does not replace the
frozen Step 1 GPU environment. Fit artifacts record the analysis package versions.

## Code fix before any activation was extracted (24 October)

The first GPU attempt stopped at the canary on every arm with "ambiguous
rendered prompt alignment", before any activation was saved. The Qwen 3.5 and
3.8 chat templates trim the user message, so the request's final newline (the
end of the trajectory close tag) is absent from the rendered prompt: the full
request occurred 0 times, the right-trimmed request exactly once, on both
tokenizers. `code_token_indices` now aligns on the right-trimmed request; code
spans end before the removed newline. A synthetic trimming test was added.
Design, layers, splits, controls and predictions are unchanged.
