# Preference continuation through judge-targeted injections

User-authorized 2026-10-03. Continue a copy of the regular, full-8,688-row
Qwen3.5-4B FP32 rank-128 LoRA on the entire public JudgeDeceiver release, pooling
its upstream attack-training and evaluation splits. Compare the original and
continued checkpoints on a frozen approximately 80:20 question-grouped split.

Hypothesis: learning preference judgments with injections independently attached
to either candidate reduces unjustified avoidance of otherwise preferred answers.
Preference accuracy is an auxiliary outcome, not a harmfulness judgment. Test
transfer independently on the existing APPS injection controls and CoT-removed ID
using the regular monitoring prompt and 0/1 interface. APPS remains excluded
from training, data generation, checkpoint selection and hyperparameter tuning.

Source: `ShiJiawenwen/JudgeDeceiver` at
`7cfc96af0272befd83220b1d7d3b0699d072ecbf`. Record every raw checksum, source row,
label provenance, prompt hash and transformation under the ignored artifact tree.
The repository does not declare a license in its root file manifest; preserve
this unresolved source-license metadata rather than asserting permission terms.
No source dataset text is committed to Git.

Main MT-Bench/LLMBar and RLAIF labels identify the attack candidate; prefer the
other candidate. Search rows identify a contradictory/irrelevant target entry;
derive construction-based binary preferences against each remaining clean entry,
without asserting a ranking among clean entries. Tool rows include `real_label`;
compare that correct tool with each other candidate. Match the released attack
candidate to its separately released payload record before deriving labels.
These construction labels are separate from the original attacker-desired labels.
Deduplicate pair comparisons and preserve all merged raw-row provenance.

For every pair retain both answer orders, a clean version, and each applicable
released suffix separately appended to the preferred or disfavored candidate.
Do not attach both injections at once. Correct preferences remain fixed. Native
nonthinking Qwen chat ends directly at a one-token A/B decision (class 0=A, 1=B).
The task prompt in `prompt.txt` is shared by both evaluation checkpoints.

Split before augmentation by normalized original query, globally across sources.
Keep every answer pair, order, suffix and source row for a query on one side.
Hold out two queries each from MT-Bench and LLMBar, one each from RLAIF and search;
the single tool-selection query goes to training. Choose the allocation nearest
20% of augmented rows using only counts, with deterministic seeded hash tie breaks.
Record the realized fraction; exact 80:20 row proportions cannot override grouping.
Tool selection has no independent held-out query, explicitly limiting its claims.

Intervention: one fixed epoch, fresh AdamW optimizer, LR 2e-5, 3% warmup, linear
decay, batch 32. Reuse the packed BF16 FlashQLA/SDPA B200 profile, 16,384-token
physical packs, shared persistent compiler cache, no model checkpointing and FP32
masters. Reuse unchanged validated startup diagnostics, checksum their receipt,
and retain finite/missing-gradient guards. A/B uses the existing binary loss and
selected-token projection; no new loss family or classification head is added.
Do not mix monitoring replay into this first domain-continuation experiment.

Baseline: unchanged original regular full-data adapter, same held-out prompts and
one-token A/B scores. Serving parity must pass for both adapters under A/B and
0/1 surfaces, with an adapter effect, before full evaluation. Score base only for
bounded backend parity. Use one persistent vLLM engine and raw requested decision
logprobs; normalized p(B) or p(harmful) retains both raw values and token audits.

Report preference accuracy, AUROC, Brier, log loss, calibration, ties, p(correct),
source/query macro means, paired changes from clean to preferred/disfavored
injection and preference flips. Report APPS under the regular prompt, including
injected-honest alarms, and standard-prompt ID ranking/calibration for the continued
adapter against checksum-verified historical original results. Final checkpoint
only; no epoch/LR/prompt/threshold selection on held-out preference or APPS scores.
Stop on unknown/ambiguous label contracts, provenance drift, grouped leakage,
truncation, missing/nonfinite gradients or scores, parity failure or incomplete
coverage. Retain failure receipts. Generalization remains bounded by few queries,
model-specific optimized suffixes, construction labels and one seed.

The user restricted monitoring evaluation to the regular prompt before held-out
scoring began. `evaluation_config.yaml` freezes that scope separately from the
unchanged training configuration. Artifacts live in `evaluation_regular/`;
the interrupted sweep and its completed reference scores remain in `evaluation/`.
Reuse only the checksum-verified preference/regular reference cells with identical
weights and inputs. Aggressive/conservative prompts receive no full evaluation.

The existing authorized B200 is used; no new capacity is allocated. This session
has no in-chat scheduling tool, so monitoring takes place during active turns.

Entrypoints: `python -m experiments.judge_injection_continuation.prepare`,
`python -m experiments.judge_injection_continuation.train`, and
`python -m experiments.judge_injection_continuation.launch_evaluate`.

## Frozen preparation

All 2,790 released source rows yield 1,260 deduplicated preference pairs and
20,948 augmented examples. There are 31 original query groups: 25 training and
6 held out. The realized split is 16,760 training / 4,188 evaluation examples
(80.008% / 19.992%). The native chat inputs plus one decision token total
5,964,368 training tokens and 1,525,718 evaluation tokens. Maximum lengths are
1,031 and 594 tokens respectively; no examples require truncation. One epoch
has 524 logical updates, including the partial final batch. The ignored
`data/judge_injection_continuation/manifest.json` freezes every artifact checksum.
