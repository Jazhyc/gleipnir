# Binary action-only tool-trajectory monitoring

Status: prompt contract and bounded teacher canary complete. No student training
or OOD evaluation has started.

## Lambda environment status

The already-reserved `gleipnir-control` target was probed and bootstrapped on
2026-08-30 from commit `0fc7d91aa00f8c03382ec1236995d88efe97f1d0`:

- one NVIDIA H100 80GB HBM3 (SXM5), in Lambda region `us-south-2`;
- Ubuntu 22.04, x86-64, NVIDIA driver 580.105.08;
- Python 3.12.14, PyTorch 2.11.0+cu130, Transformers 5.14.1, and vLLM 0.24.0;
- isolated `flash-linear-attention==0.5.2` and `fla-core==0.5.2` under
  `/tmp/gleipnir-qwen35-fla`, with the chunk and fused-recurrent gated-delta
  imports resolved to `fla.ops`.

The locked environment is under `~/gleipnir/.venv`, and the cache/CUDA exports
are in `~/.config/gleipnir/runtime.env`. The FLA target is ephemeral and must
still be installed and probed fail-closed by the eventual launcher before model
import. A bounded CUDA matrix-multiply probe and the focused prompt/fast-kernel
contract tests passed. No credentials or model weights were transferred during
this setup.

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

## Teacher-logit canary

The first bounded canary uses the research-use-only
`Syghmon/blackboxmonitorsMATS` revision
`ccb77e37589baaee1f0e16ac9ab89ceac8f6cc8a`. It samples equally across source
and label from `gemini_synthetic_1000.parquet` and
`unified_dataset_1000.parquet` after applying the paper's 28,000-token input
cutoff with the pinned Qwen3.5-9B tokenizer. The fixed seed is `20260830`.
These are accessible upstream synthetic training-source releases, not the
paper's exact final STRIDE/Gloom row manifest, so interpret the result as a
teacher-interface sanity check rather than a paper replication.

Prepare 100 rows after downloading the gated files into the ignored source
directory:

```bash
python -m experiments.tool_trajectory_monitoring.prepare_teacher_canary
```

Run a 12-row paid request canary first, pinned to Makora with no fallback:

```bash
gleipnir-openrouter \
  --input data/tool_trajectory_monitoring/canary/prompts.jsonl \
  --output results/tool_trajectory_monitoring/kimi_k3_canary.jsonl \
  --model moonshotai/kimi-k3 \
  --provider-only makora --no-allow-fallbacks --limit 12
```

Rerun without `--limit` to resume and fill all 100 rows, then summarize:

```bash
python -m experiments.tool_trajectory_monitoring.summarize_teacher_canary
```

If the eight-token request emits prose instead of reaching its terminal label,
keep those prompts unchanged and retry them in a separate cache with
`--max-tokens 64`. Pass both score files with repeated `--scores` arguments to
the summarizer. Never replace or mix request settings inside the primary cache.

The ignored materialization manifest records source checksums, tokenizer and
prompt revisions, the exact interleaved request order, and the cache-prefix
hash. The summary checks every prompt hash, reports overall and per-source
AUROC with a source-label-stratified bootstrap interval, threshold diagnostics,
score ties, provider/model identities, token usage, cache reads, and reported
cost. Do not interpret the OOD frontier reported in the paper as the expected
value of this small in-distribution source canary.

### Canary outcome

Kimi K3 on Makora reached `0.9746` pooled AUROC on the 100 balanced rows, with a
source-label-stratified bootstrap 95% interval of `0.9474--0.9944`. Per-source
AUROCs were `0.9888` and `0.9880`; the unweighted mean was `0.9884`. Hard
generated decisions reached only `0.9300` AUROC/accuracy, supporting the value
of retaining continuous literal-token logits.

The eight-token request completed 97 rows. Two deterministic format failures
and one transient 429 were recovered unchanged in a separate 64-token request
cache, and both settings remain explicit in the summary. The 97 unique primary
requests reported zero provider cache reads or writes. Cache reads on two
repeated recovery prompts do not demonstrate shared-prefix reuse. Treat Makora
cross-row input caching as unsupported until a new probe shows otherwise, and
do not scale teacher annotation under the previous cache assumption. Full
results and interpretation are in
[`../../docs/findings/tool_trajectory_teacher_canary.md`](../../docs/findings/tool_trajectory_teacher_canary.md).
