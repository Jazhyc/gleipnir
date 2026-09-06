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

## Cache preflight findings and bounded singleton follow-up

The first engine launch failed before scoring because its PATH omitted ninja.
The corrected launch matches the ID runner's executable and kernel-cache paths.
Its 24-prefix preflight demonstrated reuse (39,984 cached tokens in singleton
mode), but the combined singleton/batch gate failed. Singleton maximum error
was 0.031209; batched maximum was 0.061457, above the unchanged 0.05 cutoff.
Cold/singleton-growing/batched-growing timings were 5.113/4.037/1.856 seconds.
Do not approve the failed eight-parent configuration or claim these early-prefix
timings predict whole-campaign runtime.

Run one separate `--parallel-trajectories 1` preflight on the same examples,
retaining the MAE <=0.02 and max <=0.05 numerical limits. Only singleton
comparisons govern this variant; retain batched diagnostics as failed-context
evidence. Correct the cache-coverage criterion to require observed reuse beyond
the rubric for every source, not every short early prefix: 784-token cache
blocks cannot expose a trajectory-specific block in all those short requests.
Retain per-request hit counts so this correction is auditable. If this variant
passes, `--cache-output` may start the full unchanged 133,947-reference campaign
in the same engine at exactly one active trajectory; resume identities include
the trajectory concurrency. No numerical thresholds are loosened.

The independent singleton repeat failed (maximum error 0.093386). Batch-invariant
mode failed during initialization because vLLM does not support it for GDN_ATTN.
Neither variant started annotation. The next bounded test explicitly stores SSM
states in FP32: installed vLLM defaults these states to the BF16 model dtype.
Use the same 24 prefixes, singleton concurrency, and unchanged numerical gates;
record the dtype in the canary and resume contract. This tests recurrent-state
rounding, not teacher quality. Start full caching only if this test passes.

FP32 SSM canary also failed: MAE 0.011400464, maximum 0.120145485;
all four sources showed trajectory-specific reuse. No annotation started and
the failed process group was stopped. FP32 state storage alone is insufficient.
Upstream PR 51113 fixes align-cache poisoning mainly with speculative decoding;
this campaign does not use speculative decoding, so applicability is unproven.

An isolated `.venv-vllm028-prefix` now contains vLLM 0.28.0 with Transformers
5.14.1. The original environment and lockfile remain unchanged. Test the same
singleton canary with default SSM precision before considering annotation;
record runtime package versions in both the canary and cache resume identity.

The vLLM 0.28.0 / Torch 2.13.0 / Transformers 5.14.1 singleton test failed:
MAE 0.005888929, maximum 0.074652923, with reuse verified for every source.
No annotation started; failed workers were stopped and results collected.
Do not promote this upgrade. Before further configuration screens, add repeated
reset-cache reference scoring to separate reference variability from reuse error;
repeatedly rerunning the same gate until a pass would not establish reliability.

Repeated reset-cache scoring was exactly stable on all 24 prefixes (maximum
difference zero); reuse reproduced maximum error 0.074652923. This supports a
systematic execution-path difference, not random reference noise. The next
bounded diagnostic changes only GDN prefill from Triton to FlashInfer in the
isolated 0.28 environment, retaining repeated cold references and all thresholds.

FlashInfer passed the singleton diagnostic: MAE 0.003716594, maximum 0.030967128,
repeated cold maximum zero, and trajectory-specific reuse in all four sources.
The bounded diagnostic did not request annotation. Proceed with the same backend
and singleton concurrency through the gated full-cache entrypoint; keep the
original model revision and approved prompt. This small canary is not proof of
agreement across every long prefix, so retain per-request provenance and failures.
