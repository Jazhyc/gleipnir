# Joint monitoring and injection-resistant preference training

Authorized 2026-10-03. Train a fresh Qwen3.5-4B rank-128/alpha-256 adapter with
both the full 8,688-row regular trajectory population and the existing
JudgeDeceiver preference training pool. User selected 75% monitoring / 25%
preference examples and the regular learning rate 5e-5. The previous domain
continuation used 2e-5 and regressed ID performance; it is not the initialization.

Hypothesis: retaining monitoring supervision throughout training can acquire
injection-resistant preference judgments without the monitoring regression
observed after preference-only continuation. Forgetting and task interference
remain competing explanations; this single intervention does not isolate them.

Initialize from the same untrained zero-B LoRA used by the original regular
model. Verify every B matrix is zero and record both file and trainable-tensor
identities. The frozen base has revision
`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Use the selected packed BF16 FlashQLA/SDPA B200 profile, FP32 adapter masters,
16,384-token physical packs, logical batch 32, no model checkpointing,
fresh AdamW, 3% warmup, linear decay and one monitoring epoch. Reuse the
checksum-pinned startup receipt and persistent compiler cache. Keep
finite/missing-gradient guards; serving parity remains adapter-specific.

The task sampler visits every trajectory exactly once in seeded random order.
Every full logical batch has 24 monitoring and 8 preference examples. Preference
draws have largest-remainder quotas of 10% clean, 45% preferred-answer-injected,
45% disfavored-answer-injected, with both A/B orders balanced within each condition.
Draw without replacement within each condition/order pool until it is exhausted,
then reshuffle that pool. All 16,760 preference training examples are eligible;
one monitoring epoch draws 2,896, not a full preference epoch. Freeze the selected
row schedule and token totals before training. This yields 362 updates and
approximately 22.5% injected examples overall, instead of natural row proportions.
Sampling weights control example-level gradient mass, not token-level mass.

Monitoring keeps the unchanged regular instruction, `Prediction:` prefix, 0/1
decision logits and original cached soft teacher distributions. Preference keeps
the previously frozen quality rubric, empty decision prefix, A/B logits and hard
construction labels. Gather the correct two logits per example, including after
packed reordering. Each example contributes exactly one original binary loss:
soft BCE for monitoring or hard CE for preference. Never interpret quality labels
as harmfulness labels or manufacture preference teacher distributions. Privileged
teacher targets are attached only to monitoring rows, with explicit provenance.

Reuse the existing normalized-query 80:20 preference holdout: 25 training queries,
6 test queries, 4,188 held-out variants. All conditions and orders for a query stay
together. Preserve the original trajectory holdouts. JudgeDeceiver source,
revision, unresolved license and construction-label limitations remain those of
the [continuation experiment](../judge_injection_continuation/README.md). No raw
source text, paid responses, model weights or generated data is committed.

Freeze the final checkpoint before evaluation. Compare with checksum-verified
original regular and preference-only continued results on held-out preference,
regular-prompt APPS benchmark/honest controls, and the identical 3,012 CoT-removed
ID examples retaining assistant prose. No prompt sweep. One persistent vLLM engine
uses constrained one-token decisions and explicitly requested raw decision-token
logprobs. New adapter master/serving parity must pass on training-derived cohorts
for both A/B and 0/1, with nonzero adapter effect, before full scoring.

Report query/source macro preference accuracy, clean/injected paired flips,
ranking, calibration, thresholds, ties and APPS attack-template breakdowns.
ID source-macro pAUROC@20 is the primary retention diagnostic. APPS is excluded
from training, sampler construction and hyperparameter/checkpoint selection.
Do not promote on these exploratory reused holdouts. Extra optimizer updates,
shorter auxiliary contexts and unequal preference exposure prevent strict
compute-matched causal attribution against the original or continuation runs.

Stop on checksum/label/surface drift, query leakage, missing/nonfinite gradients,
scores, truncation, OOM, failed parity or incomplete coverage. Preserve failure
receipts. Use the already-running authorized B200; allocate no new capacity.
No in-chat heartbeat scheduler is available in this session; inspections occur
during active turns and cannot promise a follow-up after the turn ends.

Entrypoints: `python -m experiments.judge_injection_joint.prepare`, `train`,
and `launch_evaluate`. Inputs: `data/judge_injection_joint/`; outputs:
`results/judge_injection_joint/`; logs: `logs/runpod/judge_injection_joint/`.

## Frozen preparation

The schedule has 11,584 draws: 8,688 monitoring, 290 clean preference,
1,303 preferred-injected and 1,303 disfavored-injected. It visits 2,896 unique
preference variants spanning all 25 training queries, 940 pairs and 78 payload IDs.
Input tokens total 84,842,449; preference contributes 1,026,138 (1.21%).
Maximum input length is 29,337; no truncation. The one-token decisions add
11,584 tokens. The 4,188 held-out variants remain excluded from sampling.

The B200 run completed all 362 updates and 16,314 fresh evaluation scores.
Preferred-answer-injected accuracy is 100%, with 99.90% pooled preference
accuracy. ID source-macro pAUROC@20 is 0.8658 (original 0.8463; preference-only
continuation 0.7958). All injected harmless APPS controls still alarm at the
clean-code ~2% FPR operating point: preference robustness did not transfer to
the regular harmfulness-monitoring task. All six serving-parity cells passed.
The [finding](../../docs/findings/judge_injection_joint.md) records the complete
comparison, provenance, calibration and limitations.
