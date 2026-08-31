# Binary action-only tool-trajectory monitoring

Status: prompt contract, bounded teacher canary, and frozen Kimi K3 OOD baseline
complete. Student training has not started.

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

## Prompt-preparation stop condition

This preparation phase completed when the paired prompts were explicit, rendered
deterministically, preserved the full-teacher/compact-student distinction,
exposed the provider-cache boundary, and passed their contract tests. The paid
canary, OOD materialization, and Kimi baseline have since completed. Lambda
student training remains a separate phase with frozen data and run configs.

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

## Completed OOD baseline and frontier registry

The frozen Kimi K3 evaluation covers all 6,395 strict OOD rows. Its full binary
teacher prompt and immediate literal-token logits reach `0.9084` Mean-OOD
pAUROC@20 at an uncached list-price cost of `$25.1177/1K`. The run audit is in
[`../../docs/findings/tool_trajectory_kimi_ood_benchmark.md`](../../docs/findings/tool_trajectory_kimi_ood_benchmark.md).

The matched unadapted Qwen3.5-9B full-teacher-prompt baseline reaches `0.6976`
Mean-OOD pAUROC@20 at `$0.9468/1K` under the dated OpenRouter list-price
projection. It is not a frontier point. The run audit is in
[`../../docs/findings/tool_trajectory_qwen_ood_benchmark.md`](../../docs/findings/tool_trajectory_qwen_ood_benchmark.md).

The matched immediate 0--10 severity interface reaches only `0.4858`
Mean-OOD pAUROC@20 and `0.7592` Mean-OOD AUROC. A pure resolution control that
rounds the saved binary probabilities to the nearest tenth retains `0.6835`
pAUROC@20, showing that simple binning accounts for little of the generated
interface's loss. Its run audit is in
[`../../docs/findings/tool_trajectory_qwen_ordinal_ood_benchmark.md`](../../docs/findings/tool_trajectory_qwen_ordinal_ood_benchmark.md).

Adding one greedy visible analysis while keeping Qwen native thinking disabled
raises the ordinal interface to `0.6656` Mean-OOD pAUROC@20 and `0.8610`
Mean-OOD AUROC. It recovers 84.9% of the immediate ordinal pAUROC@20 deficit,
but remains `0.0320` below binary logits while costing 8.7% more at
`$1.0293/1K`. The result, boundary-recovery audit, and scope caveats are in
[`../../docs/findings/tool_trajectory_qwen_ordinal_reasoning_ood_benchmark.md`](../../docs/findings/tool_trajectory_qwen_ordinal_reasoning_ood_benchmark.md).

Keeping the same visible non-thinking analysis but extracting normalized
literal `0`/`1` logits reaches `0.6923` Mean-OOD pAUROC@20 and `0.8607`
Mean-OOD AUROC. This is `0.0054` and `0.0243` below immediate binary logits,
respectively, while costing 6.2% more at `$1.0054/1K`. AgentDojo improves
sharply, but all five other sources decline, so the matched binary-logit
ablation does not support a source-robust test-time-reasoning gain. The full
audit is in
[`../../docs/findings/tool_trajectory_qwen_binary_reasoning_ood_benchmark.md`](../../docs/findings/tool_trajectory_qwen_binary_reasoning_ood_benchmark.md).

The unadapted GPT-OSS-120B full-teacher-prompt baseline reaches `0.4789`
Mean-OOD pAUROC@20 at `$0.3270/1K` under the paper-matched OpenRouter hosted
proxy. It was technically the lowest-cost frontier endpoint but is much weaker
than the next point in the original baseline-only registry. After adding the
mixed 4B student it is no longer on the canonical frontier; the
price-sensitivity and run audit are in
[`../../docs/findings/tool_trajectory_gpt_oss_ood_benchmark.md`](../../docs/findings/tool_trajectory_gpt_oss_ood_benchmark.md).

The paper's complete cost--performance reference table and every future
completed Gleipnir OOD evaluation belong in
[`../../docs/research/tool_trajectory_ood_frontier.md`](../../docs/research/tool_trajectory_ood_frontier.md).
Follow that registry's frozen metric, cost, coverage, and provenance gates; do
not add partial canaries or silently replace rows after a price or interface
change.

## Soft-distillation OOD scaling evaluation

The completed one-epoch rank-128 soft-only campaign was evaluated on the same
6,395-row strict OOD suite under the unchanged compact student prompt. The six
matched Qwen3.5-9B tool-trajectory subsets (`204`, `504`, `996`, `2,004`,
`4,008`, and `8,688` rows) form the data-scaling curve. The 21,837-row mixed
tool-plus-prior-deception adapters for Qwen3.5-9B and Qwen3.5-4B are separate
transfer/interference tests, not extra points on the matched-data curve. All
arms use Kimi soft targets only; there is no hard-label training arm.

The frozen config is `distillation_ood_benchmark.json`. It fixes row membership,
model revisions, adapter job names, the student prompt hash, literal `0|1`
boundary, one-token vLLM interface, and the paper's raw normalized Mean-OOD
pAUROC@20 as the primary metric. Before either full lane, a 12-row source-label
canary compares the FP32 causal master adapter with its checksum-rebased vLLM
serving copy and requires Pearson correlation at least `0.99`, mean absolute
score difference at most `0.02`, and a nonzero adapter effect. The run stops on
parity failure, prompt truncation, missing logprobs, incomplete coverage, OOM,
or any backend error.

All eight full evaluations and both master-versus-serving parity gates passed.
The matched 9B curve is non-monotonic: Mean-OOD AUROC peaks at `0.9353` with
996 training rows, while pAUROC@20 peaks at `0.8365` with 4,008 rows. Adding
13,149 deception rows to the 8,688 matched rows is macro-neutral
(`-0.0015` pAUROC@20, effectively unchanged AUROC) but raises pooled AUROC by
`0.0181`. The mixed 4B adapter reaches `0.7823` pAUROC@20 and `0.9086`
AUROC at roughly one third the hosted inference proxy of mixed 9B. No OOM
occurred; four intermittent vLLM FlashInfer GDN spins were recovered from
durable evaluation shards without changing the frozen backend.

Reproduce or resume the sequential one-H100 campaign with:

```bash
python -m experiments.tool_trajectory_monitoring.run_distilled_ood_lambda
```

The dedicated scaling analysis belongs in
[`../../docs/research/tool_trajectory_distillation_ood_scaling.md`](../../docs/research/tool_trajectory_distillation_ood_scaling.md).
Only the completed full-mixed 9B and 4B comparisons are additionally registered
in the general cost--performance frontier.

## Gleipnir 4B visible-reasoning audit

The paired generation audit compares the mixed-data, soft-only rank-128
Gleipnir 4B adapter with its exact unadapted Qwen3.5-4B backbone on a frozen,
label-blind 64-row OOD sample. Both conditions use the existing visible binary
review template and one persistent vLLM engine. The run creates blinded A/B
reasoning pairs for a later judge; it is not a small-sample OOD performance
estimate. The hypothesis, sample rule, provenance, and launch command are in
[`GLEIPNIR4B_REASONING_SAMPLE.md`](GLEIPNIR4B_REASONING_SAMPLE.md).

A frozen lower-bound follow-up applies the identical mixed-data statistical
recipe to Qwen3.5-0.8B, with microbatch 1 and accumulation 32 preserving
effective batch 32. A matched longest-32 preflight was 2.95x faster than
microbatch 8 and produced byte-identical adapter hashes. It will benchmark both
the unadapted base and trained adapter on the same 6,395 OOD rows. The
hypothesis, model/tokenizer hashes,
parity gate, and stop conditions are recorded in
[`DISTILLATION_SCALING.md`](DISTILLATION_SCALING.md); this remains Kimi-soft-only
training with no hard-label arm.
