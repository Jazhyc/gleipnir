# Intermediate tool-action supervision

Status: prompt proposal awaiting user review. No prefix annotation or training
is launched by this proposal. The existing ID evaluations retain their frozen
full-trajectory prompts.

## Hypothesis and intervention

Intermediate soft teacher targets may improve credit assignment over long
trajectories. Preserve the existing Kimi K3 target for each full trajectory;
add local-teacher targets only at non-final tool-call boundaries. Do not request
targets at statement-only assistant turns, but retain those statements as
context in later prefixes. Never expose later actions, full-trajectory labels,
Kimi targets, or privileged rationales to the prefix teacher.

The proposed prompt changes are in `prompt_proposal.md`. Review and freeze the
rendered prompt and its hash before annotation. Preserve the original teacher
rubric rather than changing the meaning of problematic behavior.

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

After verified cache completion, compare against the existing monitoring-only
soft-distillation baseline using one epoch, learning rate 2e-5, and seed 0.
Normalize intermediate losses within each original trajectory so long examples
do not receive extra weight merely because they contain more tool calls.
Freeze a small loss-weight grid and selection criteria before launching runs.
Keep full-trajectory Kimi supervision distinct from intermediate local targets.
