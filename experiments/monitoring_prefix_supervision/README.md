# Intermediate tool-action supervision

Status: approved prompt materialized in `teacher_prefix.txt`; full prefix
annotation completed on Lambda with Qwen3.5-27B-FP8, TP2, vLLM 0.28.0,
FlashInfer GDN, and one active trajectory. The post-cache numerical audit failed;
the cache is not numerically validated. The user subsequently authorized using
it for an exploratory training screen despite that failure. No student training
has launched yet.
The existing ID evaluations retain their frozen full-trajectory prompts.

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

The saved FlashInfer eight-parent diagnostic did not pass: MAE 0.005613861,
maximum 0.093385874. Keep the active campaign at singleton concurrency; the
passing singleton gate is not evidence for safe eight-parent batching. Its
batched timing also includes an additional kernel compilation, so it is not a
clean throughput comparison.

The full-cache repeat preflight reproduced the passing FlashInfer result exactly.
Annotation began on 2026-09-06 around 01:07 UTC, process 152130, output
`results/monitoring_prefix_supervision/qwen35_flashinfer_cache/`, log
`logs/lambda/monitoring_prefix_supervision/qwen35_flashinfer_full.log`.
At the initial audit, 247 unique finite-logit records covered all four sources,
2,231,165 prompt tokens and 1,982,736 reused tokens, with maximum prompt length
26,959. Initial 52.8-second throughput was about 4.7 prefixes/second (roughly
eight hours if sustained, not a mature ETA). Both GPU workers were active.
This session exposes no timed agent scheduler; active continuation checks are
not a verified ten-minute scheduled heartbeat.

Training preparation now includes `gleipnir.prefix_sampling.sample_parent_prefixes`:
one uniform candidate per eligible parent, with independent seed/epoch/parent RNG
identity and sorted IDs. Selection is label-blind and unaffected by parent or
candidate ordering. Duplicate IDs fail closed. Five focused tests pass. This
helper does not remove or replace full Kimi targets; paired training integration
is still pending, and annotation continues without restarting its process.

The shared trainer now has an opt-in `student.training.prefix_loss_weight` path
under development. Each dataset row remains one parent; an eligible prefix is
tokenized separately without truncation. Prefix gradients are accumulated before
the full forward, with coefficient `lambda / ((1 + lambda) * parent_batch_size)`;
eligible full losses receive `1 / (1 + lambda)`, and prefix-free parents keep
their full weight. The existing explicit microbatch-mean accumulation policy
must remain enabled. Other auxiliary objectives and GroupDRO are disallowed for
this screen. Empty prefix materialization fails closed. CPU regression tests
are running; paired materialization, exact accumulated-gradient verification,
and the GPU preflight are still required before any training launch.

CPU validation: 59 focused trainer/sampling/loss/accumulation tests passed on
Lambda with GPU visibility disabled; seven lightweight sampling/materialization
tests passed locally. The sequential-normalization test compares gradients with
the parent-mean objective across accumulation windows 1/3/16/32 and weights
0.25/0.5. It uses a small Trainer harness, not a real Qwen GPU run.
`prepare_training.py` requires exact complete-cache coverage and hashes before
writing paired rows plus a provenance sidecar, preserving all full parent fields.
The local broader suite also passed (45 tests) after a filesystem I/O delay.
GPU preflight and campaign launch integration remain pending.

The two-arm training design is frozen in `training.yaml`: lambda 0.25 and 0.5,
one epoch, LR 2e-5, seed 0, identical sampled prefixes, all 8,688 parents.
Use the matched v6 monitoring LR/duration recipe (microbatch 1, accumulation 32,
QLoRA rank 128, selective checkpointing/compilation), rather than silently
switching back to the generic eager microbatch-8 default. Its measured throughput
support is recorded in the duration/LR experiments. Each run still requires a
largest-sequence paired GPU preflight. Select only final ID endpoints against
the existing one-epoch full-only baseline; require +0.005 macro pAUROC, no source
loss above 0.01, and no Brier regression above 0.005. No OOD tuning or automatic
extra weight search. The shared launcher now forwards the optional prefix weight;
14 launcher/materialization/sampling regression tests passed locally.

`python -m experiments.monitoring_prefix_supervision.campaign` prepares the
frozen paired-data, job, preflight-selection, and ID-evaluation manifests after
cache completion. It reuses the existing LR job factory and held-out separation
audit, keeps the unchanged full soft-target artifact, and fixes 272 optimizer
steps per arm. This is preparation only; execution orchestration and GPU
validation are still pending. Do not invoke it on the active incomplete cache.

The `run` action now reuses the duration campaign lifecycle via explicit job
factory, completed-metadata validator, and log-root hooks. Duration defaults
remain unchanged. Prefix execution requires the preserved Torch 2.11 training
environment and two idle GPUs; it does not terminate or wait behind annotation.
It performs the paired longest-row preflight, two training lanes, serving parity,
and final ID evaluation, then applies the frozen summary rule. Training metadata
must prove the prefix coefficient, parent count, accumulation policy, and pinned
kernel recipe. No GPU execution has been validated for this new objective yet.

After the cache is complete and its workers have exited:

```bash
.venv/bin/python -m experiments.monitoring_prefix_supervision.campaign prepare
.venv/bin/python -m experiments.monitoring_prefix_supervision.campaign run --revision COMMIT
```

Before paired-data preparation, a post-cache audit is now required. Its selection
is score-blind: eight longest-character prefixes from distinct parents plus eight
hash-selected additional parents per source (64 total). Character length is a
selection proxy, not an asserted token ranking. `audit_cache.py` requires a
complete cache and idle GPUs, restores the recorded serving runtime/config, and
resets the prefix cache before each fresh prediction. Numerical limits remain
MAE <=0.02 and maximum <=0.05. Training preparation independently recomputes these
errors from raw logits, checks exact sample identity and cache-contract identity,
and records the audit hash; it does not trust a summary pass flag. Eight focused
audit/materialization/sampling tests pass. The GPU audit itself awaits completion.

```bash
PYTHONPATH=src .venv-vllm028-prefix/bin/python -m experiments.monitoring_prefix_supervision.audit_cache --cache-dir results/monitoring_prefix_supervision/qwen35_flashinfer_cache
```

## Completed cache and failed full-workload audit

Annotation completed on 2026-09-06 at approximately 08:46 UTC. All 133,947
records were collected locally and passed exact reference/contract/raw-logprob
validation. Input usage totals 1,199,840,658 tokens, including 1,065,849,568
cached tokens (88.83%); annotation took 27,508 seconds. Workers exited cleanly.
The actual recorded runtime SSM cache dtype is **float32**, despite requesting
`auto`; do not describe the active 0.28 runtime as using BF16 SSM storage.

The fixed 64-prefix fresh audit then failed: MAE 0.010255635, maximum
0.120145498, with five examples exceeding the unchanged 0.05 maximum limit.
The largest disagreement was STRIDE (10,386 tokens): fresh probability
0.348645 versus cached 0.468791. Other failures include Gloom and BashArena,
and a short 1,764-token STRIDE prefix, so this is not exclusively a
longest-context phenomenon. The failed artifact is preserved as
`qwen35_flashinfer_cache/fresh_audit.json`; training remains gated off.
The audit's lingering process group was stopped after its terminal exception.

The startup canary was insufficient to establish broader numerical agreement.
Do not relax its limits, overwrite the failed audit, selectively replace only
the observed outliers, or repeat the audit until it passes. Next diagnostics
must distinguish repeatable cache-path error from fresh-reference variability
on the failed cases, preserving both original outputs and the full cache.

The five-case diagnostic found exactly repeated fresh scores, while growing
replay differed by up to 0.089178 and did not reproduce every original cached
score. This supports execution-history sensitivity rather than fresh-reference
noise. Both diagnostic artifacts remain unchanged.

The user explicitly authorized proceeding without recaching on 2026-09-06,
to test robustness to these imperfect targets. `training.yaml` pins this
exception to the exact failed-audit and cache-contract hashes. Preparation still
verifies complete coverage, identity, and raw-score provenance, recomputes the
failed numerical result, and records `fresh_audit_passed: false` plus the
authorization in the paired-data manifest. Default validation remains fail-closed.
The original two weights, full Kimi supervision, and ID selection rule are
unchanged. Results must be described as using numerically imperfect prefix
targets, not evidence that cache agreement was repaired.

## Public prediction-cache backup

Published with user approval on 2026-09-06:
https://huggingface.co/datasets/Jazhyc/Gleipnir-Prefix-Teacher-Cache
at commit `b660b46771dc152bf3b3369bdcb90311d5009e9e`.
The release includes the 133,947-record numeric cache, exact prompt, contract,
completion record, failed audit, replay diagnostic, and file checksums. It does
not include trajectory text, Kimi targets, or privileged rationales. The dataset
card prominently preserves the numerical limitation and source-access caveats.
`scripts/prepare_hf_prefix_cache.py` stages an allowlisted release after full
cache validation; `dataset_card.md` is its tracked card source.

## Evaluation restart and sharding

The weight-0.5 evaluation stalled after 512 saved rows. The owner authorized a
restart, then requested two-GPU sample sharding. `resume_evaluation.py` snapshots
all predictions saved at the handoff, skips those identities, and round-robins
whole 128-row pending batches across independent TP1 engines. Both retain the
original evaluation config hash and write isolated shard outputs. A strict merge
requires exact original coverage, no duplicate IDs, finite scores, and matching
config identities; final metrics are recomputed on the merged 3,012 rows.
Shard metrics are explicitly partial and must not be interpreted as full-ID
results. Training and the completed weight-0.25 evaluation are not rerun.
The snapshot and per-shard runtimes are retained in merge provenance.

## Final single-prefix screen result

Both 272-step, one-epoch runs completed; training took approximately 3h38m each.
The two-GPU resumed evaluation completed and merged all 3,012 ID examples with
the preserved 1,024 predictions. Final artifacts and predictions were collected
locally under `results/monitoring_prefix_training/`.

| Condition | Macro pAUROC@20 | Delta vs full-only | Macro Brier |
| --- | ---: | ---: | ---: |
| Full-only baseline | 0.871066 | — | 0.079273 |
| Prefix weight 0.25 | 0.873418 | +0.002352 | 0.079751 |
| Prefix weight 0.5 | 0.856460 | -0.014606 | 0.085594 |

Neither passes the frozen exploratory selection rule. Weight 0.25 improves
Gloom pAUROC by 0.008817 but reduces STRIDE by 0.004114. Weight 0.5 reduces
Gloom by 0.020160 and STRIDE by 0.009053; its Brier also regresses. Keep the
full-only baseline. This single-seed result does not isolate teacher mismatch,
cache numerical sensitivity, or sparse-boundary sampling as the cause. It does
not test all-boundary branching, and no OOD data were consulted for selection.
