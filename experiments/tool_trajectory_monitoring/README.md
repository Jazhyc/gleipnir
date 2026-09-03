# Binary action-only tool-trajectory monitoring

Status: prompt contract, Kimi K3 and K2.6 OOD baselines, student training, and
frozen student OOD evaluation complete.

## Lambda environment status

The current long-lived `gleipnir-improvement` target was probed on 2026-09-03:

- two NVIDIA H100 80GB HBM3 GPUs (SXM5), in Lambda region `us-south-2`;
- Ubuntu 24.04, NVIDIA driver 580.126.20, system CUDA toolkit 13.0;
- the locked Python 3.12.14 environment with PyTorch 2.11.0+cu130;
- pinned FLA 0.5.2 plus a source-built `causal-conv1d==1.6.2.post1` in isolated
  `/tmp` targets.

Both GPUs passed causal-convolution FP32/BF16 forward/backward checks. A Qwen3.5
4B-width, 8,192-token BF16 training microbenchmark measured a 2.17x convolution
speedup and a 1.60x gated-delta-sublayer speedup over Transformers' exact Torch
convolution fallback. These are kernel and sublayer measurements, not a
whole-model throughput estimate; retain the longest-sequence training preflight
before changing projected campaign duration. The mixed-4B launcher now installs
and probes both fast paths fail-closed and records the bound implementation in
training metadata.

The following `gleipnir-control` details are retained as the historical
environment for the completed campaign; that target was terminated on
2026-09-01.

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
- [`prompts/reasoning_judge_system_v1.txt`](prompts/reasoning_judge_system_v1.txt)
  and [`prompts/reasoning_judge_user_v1.txt`](prompts/reasoning_judge_user_v1.txt)
  are the separate frozen contract for independently judging the visible
  reasoning sample after its passed paid canary. They are not part of the
  teacher/student prompt manifest.

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

The completed Kimi K2.6 Inceptron follow-up reaches `0.919404` Mean-OOD
pAUROC@20 and `0.975107` Mean-OOD AUROC on all 6,395 rows. It reuses the same
full binary prompt, disables reasoning, and extracts terminal literal-token
logits through a provider-pinned OpenRouter route. Its exact uncached cost is
`$7.9110/1K` at the paper's Kimi-K2.6 token prices, making it both stronger and
68.5% cheaper than the completed Kimi K3 point. The Inceptron operational rate
would instead be `$4.4182/1K`. The exact hypothesis, canary, per-source result,
cost, and serving audit are in
[`KIMI_K26_OOD_BENCHMARK.md`](KIMI_K26_OOD_BENCHMARK.md).

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

## Gleipnir 4B held-out in-distribution evaluation

Before training-hyperparameter selection, evaluate the completed mixed-data
Gleipnir 4B adapter and its unadapted Qwen3.5-4B backbone on all 3,012 frozen
held-out rows: 946 STRIDE-test and 2,066 Gloom-Exfiltration trajectories. The
hypothesis is that the adapter improves source-macro direct-logit ranking under
the same compact student prompt used for deployment. The primary metric is the
unweighted two-source macro normalized pAUROC@20; report source-level and macro
AUROC, calibration, threshold diagnostics, and ties as well. Do not select on
the pooled metric alone because the larger Gloom-Exfiltration source would
dominate it.

The unadapted backbone is the baseline. This run evaluates one already-completed
checkpoint and performs no model selection. Stop on any row, prompt, model,
adapter, or checksum drift; a failed four-row source/label-balanced canary; a
missing decision-token logprob; an OOM/backend failure; or any non-finite score.
The six-source strict-OOD suite remains outside this evaluation. The frozen
contract is `gleipnir4b_id_benchmark.json`, and the compact-prompt materializer
is `prepare_distillation_id.py`.

The backend preflight compared vLLM's explicit Triton and FlashInfer GDN
prefill paths on the same H100 SXM5 and balanced four-row canary. Triton reduced
cold engine initialization from 364.08 to 119.80 seconds, and its adapter scores
agreed with FlashInfer within `2e-8`. However, Triton's unadapted-base scores
failed the predeclared eager-parity mean-error ceiling (`0.020512 > 0.020000`),
while FlashInfer passed (`0.014076`). Freeze FlashInfer for the official run to
preserve the established numerical contract; the systems result does not
license changing the evaluation backend after observing full-set scores.

## Gleipnir 4B visible-reasoning audit

The completed paired generation audit compares the mixed-data, soft-only rank-128
Gleipnir 4B adapter with its exact unadapted Qwen3.5-4B backbone on a frozen,
label-blind 64-row OOD sample. Both conditions use the existing visible binary
review template and one persistent vLLM engine. All 64 blinded A/B reasoning
pairs are ready for a later judge; no reasoning-quality conclusion has yet been
drawn, and this is not a small-sample OOD performance estimate. The hypothesis,
sample rule, generation audit, provenance, and launch command are in
[`GLEIPNIR4B_REASONING_SAMPLE.md`](GLEIPNIR4B_REASONING_SAMPLE.md).

## Gleipnir 4B full-specification prompt ablation

The compact student prompt may under-elicit monitoring knowledge that the
soft-distilled adapter learned. The frozen full-specification ablation changes
only the prompt supplied to the completed mixed-data Gleipnir 4B adapter; it
retains the same 6,395 strict-OOD rows, checkpoint, non-thinking chat boundary,
literal `0|1` logits, and metric implementation as the compact-prompt result.
The existing `0.7823` Mean-OOD pAUROC@20 and `0.9086` Mean-OOD AUROC are the
matched baseline. The intervention, baseline, held-out rule, and fail-closed
conditions are recorded before evaluation in
`distillation_mixed_qwen4b_teacher_prompt_ood_benchmark.json`.

The completed full-prompt run reaches `0.782758` Mean-OOD pAUROC@20 and
`0.907718` Mean-OOD AUROC, changes of only `+0.000409` and `-0.000865` from
the compact-prompt baseline. The richer instruction changes individual scores
(Pearson `0.984396`, 217 threshold flips) but does not improve the
source-balanced aggregate. It also adds 9.38% prompt tokens. See
[`../../docs/findings/tool_trajectory_gleipnir4b_full_prompt_ood_ablation.md`](../../docs/findings/tool_trajectory_gleipnir4b_full_prompt_ood_ablation.md).

## Qwen3.5-27B compact-rubric OpenRouter ablation

The frozen hosted-base ablation changes the completed Qwen3.5-27B full-rubric
condition to the compact student rubric on the same 6,395 strict-OOD rows. It
uses OpenRouter's dated `qwen/qwen3.5-27b-20260224` route pinned to Alibaba with
fallbacks disabled. Each request keeps the rubric unchanged as the sole user
message and allows at most eight tokens to reach its required terminal
`Prediction:0|1` string. Only the final digit's `0`/`1` top logprobs are scored;
any rationale or extra prose is rejected. Alibaba and Novita both reject raw
completions and close assistant-prefill messages, so the fixed prefix is
generated rather than prefilled. A full-rubric parity canary must therefore
match the saved local scores before the compact run.
The projected full cost is `$10.87`, with a frozen `$12.00` stop ceiling.

Before the full run, a balanced 12-row full-rubric canary must match the saved
local 27B logits at Pearson `>=0.95` and mean absolute difference `<=0.10`; a
separate compact-rubric canary must satisfy the terminal-token contract. The
frozen config is `qwen27b_compact_openrouter_ood_benchmark.json`.

The Alibaba canaries passed: all responses were exactly three tokens ending in
the binary digit, the final cache-identity full-rubric scores reached Pearson
`0.957254` and mean absolute difference `0.044271` against local vLLM, and all
12 compact-rubric rows returned both literal-token logprobs without retries.
This licenses the resumable full run; it is not evidence about OOD performance
by itself.

The full campaign was paused by the user on 2026-09-01 after OpenRouter's
Alibaba route repeatedly returned upstream shared-pool `429` and rate-ramp
`502` errors. The durable compact cache contains 5,787 of 6,395 unique finite
rows, all attributed to Alibaba, with SHA-256
`041254ed18918cd55b7cbcae604c9d258cc842014756707d1565ba9e1fed875f`.
The separate 12-row parity cache has SHA-256
`0aa867afd61e873e34ca20886942d1c87159c6177c754b11d7a0385de36c4acc`.
No aggregate result is reported from the incomplete cache. The ignored
`pause_state.json` beside the cache records the exact restart command and
durable provider-reported spend.

The Qwen3.5-0.8B lower-bound follow-up was intentionally stopped after systems
preflight and 12 of 683 optimizer steps. QLoRA microbatch 1 was faster than both
microbatch 8 and BF16 LoRA, but the projected 2.8--3.3-hour run was only about
1.7--2.0x faster than the completed 4B campaign. No full checkpoint or OOD
metrics exist, and the attempt is excluded from the scaling curves and frontier.
The frozen recipe and stop rationale are recorded in
[`DISTILLATION_SCALING.md`](DISTILLATION_SCALING.md) and the corresponding
[systems finding](../../docs/findings/tool_trajectory_qwen08b_stopped_systems_preflight.md).
