# Intermediate tool-action supervision

Status: prompt wording approved by the user; full instruction materialized in
`teacher_prefix.txt`. No prefix annotation or training has launched yet. The
existing ID evaluations retain their frozen full-trajectory prompts.

## Hypothesis and intervention

Intermediate soft teacher targets may improve credit assignment over long
trajectories. Preserve the existing Kimi K3 target for each full trajectory;
add local-teacher targets only at non-final tool-call boundaries. Do not request
targets at statement-only assistant turns, but retain those statements as
context in later prefixes. Never expose later actions, full-trajectory labels,
Kimi targets, or privileged rationales to the prefix teacher.

The reviewed changes are recorded in `prompt_proposal.md`; `teacher_prefix.txt`
contains the complete instruction. Freeze the rendered prompt and its hash
before annotation. Preserve the original teacher rubric rather than changing
the meaning of problematic behavior.

## Selection and execution gates

Complete the three authorized FP8 ID screens. Select by source-macro pAUROC@20;
when models are within 0.001 of the best, prefer measured faster prefix scoring,
not assumed MoE speed. Report per-source metrics and calibration alongside the
selection. Do not tune on OOD data.

Before scaling annotation, audit source-specific tool-call boundaries and counts,
exclude the final full-trajectory boundary, and verify prefix-cache reuse with
measured cache hits and matched cached/uncached score and timing canaries.
Use a persistent engine and trajectory-grouped requests. Append the decision
suffix independently to each prefix; previous monitor predictions must never
become part of the next trajectory. Record model revision, prompt and request
hashes, raw decision logprobs, normalized targets, token usage, timestamps,
failures, and resume identity. Stop on leakage, invalid boundaries, missing or
nonfinite logits, provenance drift, or failed cache correctness checks.

## Subsequent training proposal

The local candidate-boundary audit is reproducible with
`python -m experiments.monitoring_prefix_supervision.audit`. It currently finds
133,947 non-final tool-bearing candidate prefixes: STRIDE 26,791, BashArena
72,894, BashBench 27,440, and Gloom 6,822. CoT-Red-Handed has none. Excluding
statement-only turns removed 201 candidates; 12 additional missing step closes
are unresolved and receive no guessed boundary. These are
not yet validated annotation counts: 822 BashArena rows trigger conservative
delimiter/step warnings and need format review. The audit never calls a teacher
or silently excludes flagged rows from a launched cache.

BashArena step numbers can jump across `<middle_summary>` sections. Ending a
prefix at the next step opening would leak a later-events summary into the
preceding action target. Candidate endpoints now use the current `</step_N>`
instead; the original intervening summary remains context for later actions.
Seven focused tests include this leakage case and missing-close handling.

Preparation completed with 133,947 references under
`data/monitoring_prefix_supervision/`. The references, original student rows,
and approved instruction have been transferred to Lambda and their SHA-256
hashes verified there against the local manifest. No teacher outputs exist yet.
The remaining 12 malformed step boundaries are explicitly excluded, not guessed;
other valid boundaries in those trajectories retain the unchanged visible text.

`cache_canary.py` is staged for a bounded, GPU-idleness-guarded preflight after
model selection. It scores three growing prefixes from two trajectories in each
of four sources, comparing reset-per-request against singleton and eight-parent
batched reused-prefix execution in the same engine.
Require mean probability error <=0.02, maximum <=0.05, and cache hits beyond
the shared instruction on every second/third prefix. Record cold and reused
timings before estimating full-campaign throughput. This implementation has
not yet passed a GPU preflight; merely enabling prefix caching is not evidence
that it works. The code explicitly requests `mamba_cache_mode=all` and Triton
GDN, based on the installed vLLM 0.24.0 cache configuration.

After verified cache completion, compare against the existing monitoring-only
soft-distillation baseline using one epoch, learning rate 2e-5, and seed 0.
Normalize intermediate losses within each original trajectory so long examples
do not receive extra weight merely because they contain more tool calls.
Freeze a small loss-weight grid and selection criteria before launching runs.
Keep full-trajectory Kimi supervision distinct from intermediate local targets.

The full-cache path (`cache.py`, invoked with the canary's `--cache-output`)
advances up to eight independent trajectories one prefix at a time, appends raw
decision logprobs with timestamps and token/cache-hit counts, and fsyncs each
completed batch. Resume checks the frozen model/prompt contract, reference
identities, duplicates, and normalized probabilities against raw logprobs.
Malformed/truncated JSONL tails fail closed and require explicit recovery;
completed valid rows are not regenerated. Mocked end-to-end tests cover
multi-cohort scheduling, completed-cache reuse, and partial-cache resumption.
This full path is not yet GPU-validated or launched.

For the subsequent small training screen, the proposed per-trajectory objective
is `(L_Kimi_full + lambda * mean(L_local_prefix)) / (1 + lambda)`, with lambda
0.25 and 0.5 alongside the unchanged full-only baseline. Rows without eligible
prefixes retain their full loss without downweighting. Freeze the sampling and
optimizer-step budget before launch: expanding 134K prefixes into independent
training rows would otherwise change both trajectory weighting and training
duration. Uniformly sampling one prefix per parent supplies an unbiased estimate
of its mean prefix loss, but the sampling implementation still needs validation.

`gleipnir.prefix_loss.trajectory_prefix_loss` implements this reduction on
unreduced loss vectors. Four tests verify parent weighting, exact gradient
scales, invariance to repeated identical prefixes, empty-prefix behavior,
zero prefix weight, and invalid parent indices. It is not yet integrated into
the training collator/forward path; no training run has been launched.
