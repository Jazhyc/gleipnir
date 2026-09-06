# Matched Qwen/Kimi full-trajectory agreement

## Qwen3.8-Flash matched comparison protocol (2026-09-07)

User requested `qwen/qwen3.8-flash` on the same matched samples, conditional on
direct-logprob support. Alibaba is the sole catalogued endpoint, advertising
logprobs/top_logprobs; quantization is unspecified. Hypothesis: the new flash
teacher may improve inexpensive ranking/calibration over the candidates tested.
Use `--config experiments/teacher_agreement/qwen38flash.yaml` with the existing
`experiments.teacher_agreement.glm` prepare/canary/run/analyze entrypoint.
One original full-prompt canary must return actual terminal 0/1 logprobs and
zero reasoning tokens before completing all 640 rows. Preserve the existing
rubric, binary normalization, temperature zero, eight-token cap, reasoning none,
eight concurrent workers, two bounded transport retries and no fallbacks.
Freeze all matched inputs and prior MiniMax/K2.6 caches; no new baseline calls.
Price caps $0.15/M input and $0.47/M output imply roughly $1.05 input using
the Qwen3.5 token estimate, before tokenizer/cache/retry differences. No explicit
cache savings assumed. Stop on unsupported direct scoring or incomplete/invalid
records; do not substitute reasoning, hard labels or elicited probabilities.
Report all five teachers' pooled/source AUROC, ECE(5/10/20), Brier, log loss,
ties and threshold diagnostics. Balanced training-population diagnostic only;
no calibration fitting, student/prefix jobs, GPU use, or ID/OOD selection.

The one-row canary passed, but the larger pass returned free-form explanation
on at least one row and upstream 429s. Stopped the initial process (PID 1670236)
without imputing failed scores or reporting a selected-subset metric.
The next bounded interface canary uses an assistant `Prediction:` prefill
(`partial: true`) and a one-token cap with scalar 0/1 extraction, in a separate
`qwen38flash_alibaba_prefill` root and `qwen38flash_prefill.yaml` config.
This changes chat serialization explicitly, analogous to local Qwen's prefilled
decision boundary, but preserves the original user rubric. No old scores may
be mixed into this version. Require 10 source/label canaries before any full
prefill pass. Provider documentation describes Partial Mode; actual OpenRouter
support must be established empirically. Stop if prefill or alternatives fail.

## Kimi K2.6 matched comparison protocol (2026-09-07)

User authorized Kimi K2.6 through Inceptron on the same 640 final-boundary
prompts. Hypothesis: the older Kimi teacher may provide a better inexpensive
ranking/calibration tradeoff than Qwen or MiniMax. Reuse the same config-driven
runner with `--config experiments/teacher_agreement/kimi26.yaml` and phases
`prepare`, `canary --limit 1`, `run`, `analyze`. Freeze the original prompt/pair
hashes and the existing MiniMax cache; report all four teachers on exact IDs.
Pin `inceptron/int4`, reasoning none, temperature zero, 8-token cap, top-5
logprobs, eight workers, no fallbacks or explicit cache assumption. Require
both real decision alternatives and zero reported reasoning tokens. Cache
resumes the canary in place; settings and prompt hashes must match.
Catalog caps: $0.56/M input, $3.39/M output; approximately $3.94 input using
7.03M Qwen tokens as a proxy, before tokenizer differences/cache hits/retries.
Stop on incomplete or invalid scoring; do not switch to reasoning or tune the
prompt. Report pooled and per-source AUROC, ECE(5/10/20), Brier, log loss,
ties and threshold diagnostics. This is evaluation-only on a balanced training
sample, with no calibration fitting, ID/OOD promotion, student runs or GPU use.

Completed 640/640 with zero reasoning tokens. Six HTTP-429 failures from the
concurrency-eight pass were completed with `run --concurrency 2`; the override
only changes transport parallelism, not the frozen request settings or scores.
K2.6 pooled AUROC 0.960449, ECE(10) 0.051102, Brier 0.081665; reported
successful-row charges $3.380963. See
`docs/findings/kimi26_matched_teacher_comparison.md` for all four teachers.

## MiniMax full comparison protocol (2026-09-07)

User authorized completing all 640 matched rows after the direct-logit canary.
Hypothesis: MiniMax M3 may improve teacher ranking/calibration over local Qwen.
Reuse the existing runner with `--config experiments/teacher_agreement/minimax.yaml`
and phases `prepare`, `run`, `analyze`; the historical module name is `glm`.
Resume the original canary cache in place (its filename retains `canary`), with
the same settings hash. Freeze original prompts, matched labels/scores and config.
Pin CoreWeave FP4, require reasoning none and zero reported reasoning tokens,
8 output tokens, true terminal 0/1 logprobs, no fallback or prompt changes.
Eight concurrent requests; bounded two retries for transport/server failures.
Stop on incomplete scoring or identity/logprob drift; no reasoning fallback.
Report pooled and source AUROC and ECE (5/10/20 bins), Brier/log loss,
confidence/accuracy, ties and threshold diagnostics against existing labels.
This is a source/label-balanced training diagnostic, not held-out promotion.
No calibrator, student training, new prefix annotations or GPU use is authorized.
Expected input cost around $1.6 using the existing Qwen-token estimate; actual
MiniMax tokenizer usage, cache hits and retries can change the bill.

Completed all 640 rows with zero reasoning tokens and valid paired logprobs.
MiniMax pooled AUROC 0.942275, ECE(10) 0.078941, Brier 0.097374; total reported
cost $1.416517. See `docs/findings/minimax_matched_teacher_comparison.md` for
the matched three-teacher comparison, source diagnostics and caveats.

## MiniMax direct-logit capability check (2026-09-07)

The user replaced the GLM investigation with a MiniMax M3 non-thinking
capability check. No GLM batch was launched. A single original matched prompt
(`342b86151aa2f9429bf98f33`) succeeded through `minimax/minimax-m3`, pinned to
`coreweave/fp4`, with reasoning effort none, temperature zero, max_tokens 8,
top_logprobs 5, no fallback, no explicit caching and evaluation-only filtering.
The response was `Prediction:0`, 4 completion tokens, **0 reasoning tokens**,
25,748 input tokens, 1.339 seconds client latency and reported cost $0.0058666212.
Actual decision logprobs: 0 = -0.029780270531773567;
1 = -3.529780387878418; binary-normalized positive score = 0.02931222741248963.
Artifact: `results/teacher_agreement/minimax_m3_coreweave_canary.jsonl` retains
prompt/settings hashes, raw alternatives, model/provider identity and usage.
This establishes endpoint capability for one sample, not a completed 640-row
AUROC/calibration comparison. CoreWeave's FP4 quantization must be disclosed.
Current endpoint price caps: $0.23/M input, $0.96/M output. No full MiniMax
batch was launched during this capability check.
The already-in-flight 4,096-token GLM canary finished with one valid score
before the attempted cancellation reached it; its artifact is preserved.
No additional GLM requests were started after the user's switch to MiniMax.

## GLM 5.3 Flash follow-up protocol

User-authorized 2026-09-07: evaluate `z-ai/glm-5.3-flash` through OpenRouter,
pinned to Wafer without fallbacks, on the same 640 original detailed prompts.
Hypothesis: this inexpensive teacher may offer a better ranking/calibration
tradeoff than Qwen. Reuse Kimi and Qwen scores; do not annotate new prefixes,
train a student, fit a calibrator, or consult ID/OOD selection sets.
Config: `glm.yaml`; entrypoint: `python -m experiments.teacher_agreement.glm`.
Use the existing resumable OpenRouter client, temperature zero, reasoning none,
8-token cap and terminal literal binary logprobs, normalized over 0/1.
This matches the Kimi API output contract; chat serialization and quantization
remain model/provider differences. Wafer catalog quantization is unknown.
Run one canary first, then ten (one per source/label cell); fail closed on
missing decision alternatives, routing/identity drift or incomplete scoring.
Both candidate label probabilities must be returned, not fabricated from a
hard response. Preserve raw logprobs, prompt/settings hashes and usage. Price
caps are $0.10/M input and $0.35/M output; expected input cost is about $0.70
using 7.03M Qwen tokens as a proxy. No explicit cache savings are assumed.
Report pooled and per-source AUROC, equal-width ECE (5/10/20 bins), Brier,
log loss, confidence/accuracy, ties, threshold diagnostics and actual usage.
The comparison is descriptive on a balanced training sample, not promotion
evidence. Stop after one complete 640-row pass; no adaptive prompt tuning.

Initial canary was rejected before inference by OpenRouter's distillable-text
filter: GLM is not marked as permitting distillation. This follow-up therefore
explicitly disables that filter for **evaluation only**. Do not reuse its
outputs as student targets without a separate licensing/terms review. The
shared client's default remains to require distillation permission.

The subsequent reasoning-disabled canary was also rejected before inference:
Wafer requires reasoning for GLM. Revised operational diagnostic uses minimal
reasoning with a bounded 512-token total completion budget in a separate
`glm53_flash_wafer_minimal` artifact directory. This is **post-reasoning** binary
calibration, not matched non-thinking logits. Original rubric and terminal
decision-token scoring remain unchanged. Do not interpret differences as an
isolated model-quality comparison. If this cap cannot produce a valid terminal
binary score, stop before the full pass rather than silently increase compute.

The 512-token request exhausted its entire budget on hidden reasoning and
returned no decision/logprobs (reported cost $0.002701809). The user subsequently
asked to continue; a new bounded canary uses minimal reasoning, 4,096 tokens,
and a separate `glm53_flash_wafer_minimal4096` directory. At the configured
rates, 640 requests at the full output cap add at most about $0.92 in completion
charges before retries, in addition to input. Do not launch the full pass
unless the new canaries provide both actual binary logprobs.

Licensing clarification: the model-weight repository currently carries the MIT
license (`https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/LICENSE`).
OpenRouter calls its distillation metadata best-effort and advises checking
actual terms (`https://openrouter.ai/docs/cookbook/evaluate-and-optimize/distillation`).
The failed allowlist check does not establish an author prohibition. This run
remains evaluation-only as requested; no training permission conclusion is drawn.

Hypothesis: confidence-scale and/or ranking differences between the local Qwen
prefix teacher and Kimi full-endpoint teacher may introduce conflicting student
supervision. Compare the same full trajectories and original detailed teacher
rubric; do not confound teacher identity with the visible trajectory length.

Freeze 64 rows per source/label cell (640 total across five training sources),
ordered by SHA256 of seed and ID, unique trajectory hashes within the sample.
Selection never reads teacher scores. Match every Kimi target by exact ID,
dataset, label and rendered-prompt SHA; retain original raw decision logprobs.
No new Kimi calls, ID/OOD tuning, target rescaling or student training.

Reuse the proven Qwen3.5-27B-FP8 TP2 non-thinking teacher evaluator on both H100s,
original Torch2.11/vLLM0.24 environment, prefix caching disabled. Singleton
scoring after the existing repeated-reference canary minimizes batching changes.
Retain the pinned model revision, raw decision logprobs, one-token interface,
prompt hashes and resumable records. Stop on provenance/context drift, invalid
scores, failed canary or GPU errors. Do not interrupt unrelated GPU work.

Report pooled and per-source probability MAE, signed Qwen-minus-Kimi bias,
confidence, binary-decision disagreement, Spearman rank agreement, and descriptive
log-odds slope/intercept. Raw logprobs across vocabularies are not directly
comparable; normalize over the same binary outcomes and compare log-odds margins.
Label-based Brier scores are descriptive on the sampled training population,
not held-out model-selection evidence. Source/label balance changes population
weights, and numerical FP8/interface differences remain part of this comparison.
Differences cannot by themselves establish why prefix student training failed.
Also stratify by the original Kimi provider because the cache spans three
providers with small, previously measured numerical differences.

Entrypoint: `python -m experiments.teacher_agreement.run prepare|run|analyze`.
Preparation consumes `config.yaml`, freezes prompt/Kimi/config artifacts under
`results/teacher_agreement/`, and audits exact Qwen token lengths. Run reuses the
shared evaluator and then computes agreement; never calibrates targets in place.

## Execution

2026-09-06: frozen sample passes the shared evaluator's input/config validation:
640 rows, 7,029,373 Qwen prompt tokens, maximum 30,149. Both reserved H100s
were idle before launch. Lambda supervisor PID 167521 uses the original `.venv`;
log: `logs/lambda/teacher_agreement/run.log`. Three focused statistics tests and
Ruff checks pass. Final numerical canary and agreement results remain pending.

Completed the same day: all 640 unique IDs passed local re-analysis, including
exact source-prompt hashes, labels, dataset identities, frozen config SHA and
raw-logprob-to-probability consistency. Numerical canary passed with mean
absolute probability difference 0.002924 and maximum 0.014350. Final results:
`docs/findings/matched_qwen_kimi_teacher_agreement.md`. Both teacher inputs and
Qwen outputs are retained locally under the ignored result directory.

Calibration follow-up (2026-09-07): run
`PYTHONPATH=src .venv/bin/python -m experiments.teacher_agreement.calibration`
to compute unmodified-score Brier/log loss, positive-class equal-width ECE
(5/10/20 bins), confidence/accuracy, source breakdowns and reliability diagrams.
This is a descriptive label-based diagnostic on the existing balanced sample;
it fits no correction and makes no new teacher calls. Bin event-rate intervals
are Wilson intervals assuming independent samples, not grouped confidence bounds.
