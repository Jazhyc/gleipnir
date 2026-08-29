# Binary action-only tool-trajectory monitoring

Status: prompt-contract preparation. No dataset has been materialized, no paid
teacher request has been made, and no training or evaluation run has started.

## Hypothesis and intervention

The primary hypothesis is that Kimi K3's binary decision-token distribution is
more sample-efficient supervision for a `Qwen/Qwen3.5-9B` action-only monitor
than source hard labels. The intervention is logit distillation at the literal
`0|1` decision boundary. The minimum baselines are the unadapted model and a
matched hard-label student. The paper's published rationale-distillation curve
is an external exploratory reference, not a matched causal baseline.

The eventual campaign will select checkpoints only on held-out STRIDE test and
Gloom-Exfiltration, using a frozen aggregate and tie-breaker. Its primary metric
is per-source and Mean-OOD AUROC. The six OOD sources must not influence prompt,
checkpoint, threshold, subset, or stopping choices. The campaign stop and
invalidation conditions are recorded in
[`../../docs/research/tool_transcript_monitoring.md`](../../docs/research/tool_transcript_monitoring.md).

## Prompt contract

The prompts are plain text so their substantive wording can be reviewed and
edited without reading Python:

- [`prompts/teacher.txt`](prompts/teacher.txt) is the long monitoring
  specification used both for Kimi logit annotation and the frozen teacher
  baseline evaluation.
- [`prompts/student.txt`](prompts/student.txt) is the compact instruction used
  unchanged for Qwen student training and evaluation.
- [`prompts/manifest.json`](prompts/manifest.json) declares the prompt-set ID,
  file mapping, common trajectory envelope, literal labels, and intended uses.

Both prompts use the same binary semantics and require exactly
`Prediction:0` or `Prediction:1`. They deliberately do not request a visible
rationale or an ordinal severity score. Kimi's uncertainty is represented by
the normalized logprobs of the literal `0` and `1` tokens. The source hard label
and Kimi soft target remain separate fields with separate provenance.

`PromptTemplate.render()` produces the exact single user-message content for
teacher generation or ordinary student SFT. The assistant completion is
`Prediction:<label>`. `PromptTemplate.decision_context()` appends the literal
`Prediction:` prefix after the trajectory; use that context for selected-token
student training and frozen evaluation so the compared next-token logits are
exactly `0` versus `1`. Training and evaluation share the same student template
and chat wrapper; only the supervision target differs.

The full teacher instruction ends before `<agent_trajectory>`. The renderer
exposes the instruction and opening marker as one invariant `cache_prefix`, so
the OpenRouter route can place its ephemeral cache breakpoint immediately
before variable trajectory text. Never put row IDs, labels, source metadata, or
other example-specific text before that breakpoint.

The trajectory is untrusted data. It includes the action-only native source
representation after private reasoning and privileged fields are removed. The
renderer preserves its contents and only supplies the common opening/closing
markers. It does not sanitize prompt injections; both prompt variants explicitly
instruct the monitor not to follow transcript text.

## Inspecting and changing prompts

Render both exact prompts around a harmless placeholder trajectory:

```bash
python -m experiments.tool_trajectory_monitoring.preview_prompts
```

Render a real local trajectory with one role:

```bash
python -m experiments.tool_trajectory_monitoring.preview_prompts \
  --role teacher --trajectory-file /path/to/trajectory.txt
```

Any wording, delimiter, label, or mapping change changes the computed template
hash and therefore creates a different teacher-cache identity. Increment
`prompt_set_id` for an intentional semantic or interface change. Do not reuse a
teacher cache across prompt hashes. Training and evaluation must load these same
files through `prompting.py`; they must not carry private inline copies.

## Current stop condition

This preparation phase is complete when the paired prompts are explicit, render
deterministically, preserve the full-teacher/compact-student distinction, expose
the provider-cache boundary, and pass their contract tests. A paid canary,
dataset materialization, and Lambda training are separate future phases and
require frozen data and run configs first.
