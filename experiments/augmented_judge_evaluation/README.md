# Augmented adapters on the original JudgeDeceiver holdout

User requested 2026-10-04: evaluate the completed monitor-directed augmented
Qwen3.5-4B and Qwen3.5-9B adapters on the original auxiliary task's held-out
examples, comparing with unaugmented regular adapters. Use the regular student
prompt. The user clarified that this means the original auxiliary A/B
quality-judging instruction, to measure avoidance of injected answers.

Hypothesis: monitoring augmentation reduces injection-driven suspiciousness
or avoidance on independently sourced JudgeDeceiver payloads. Keep clean,
preferred-answer-injected and disfavored-answer-injected comparisons paired.
Use only the fixed final adapters, with no training, checkpoint selection,
prompt sweep or held-out threshold optimization.

Frozen source: `data/judge_injection_continuation/test.jsonl`, 4,188 variants
(19.992% of the prepared population) covering 252 pairs and six query groups.
All orders and suffixes for a query remain on the original split. Construction
preference labels are not harmfulness labels. The augmented monitoring runs
did not train on this auxiliary dataset. Source revision and unresolved license
remain as documented in `../judge_injection_continuation/README.md`.

The original 4B A/B preference scores are cached under
`results/judge_injection_continuation/evaluation_regular/original/`.
No cached 9B JudgeDeceiver score population was found. Reuse the checksum-verified
4B regular results and score augmented 4B, base 9B, regular 9B and augmented 9B
freshly. The user explicitly added a full base-9B pass before scoring began.
No new harmfulness evaluation or inference-prompt sweep is introduced.

Require checksum-pinned inputs/checkpoints, zero truncation and full unique
coverage. Each new model/adapter/surface combination must pass a bounded
master-to-serving parity comparison on training-derived canaries with finite
scores and nonzero adapter effect before persistent-vLLM population scoring.
Use constrained one-token responses and requested raw decision-token logprobs.
Use one engine per backbone and batch adapter conditions within that engine.

Report clean/injected paired score shifts and flips, equal-query/source
breakdowns, calibration/ranking for the A/B target where applicable, and score
ties. For harmfulness scoring, preserve preference labels separately and do not
report harmfulness accuracy/FPR from those labels. Six queries are the effective
task diversity, not 4,188 independent tasks. Stop on input/checkpoint drift,
nonfinite or missing scores, truncation, OOM, failed parity or incomplete coverage.

Infrastructure: the existing B200 host could not restart and US-NC-2 had no
capacity. New pod `aqmiipyogkjgi3` runs in EU-RO-1 at $6.79/hour, with persistent
100 GB standard network volume `cb7bxu71ug` ($7/month). GPU probe confirmed a
B200 with 183,359 MiB, driver 580.126.20 and CUDA 13.0, zero active processes
and zero volatile uncorrected ECC errors. The original US volume is preserved.
No in-chat scheduler is available; active-turn checks cannot promise later
agent follow-ups.

Automatic approval review rejected whole-repository sync because explicit
authorization for exporting private code was absent. No repository payload
was transferred. The scoped, checksummed transfer proposal is
`results/augmented_judge_evaluation/transfer_plan.json`. The user subsequently
explicitly authorized transferring whatever data is needed to Runpod; code,
frozen holdout, canaries, adapters and cached baseline scores are now authorized.
The initial rejection is preserved as an infrastructure receipt, not a numerical
failure. No result or parity pass is claimed before actual checks finish.

Entrypoints: `python -m experiments.augmented_judge_evaluation.prepare` locally,
then `python -m experiments.augmented_judge_evaluation.launch` on the B200 after
the locked environment and frozen artifacts are transferred. Work is on `main`
at the user's explicit request. The small A/B
population stays within a 4,096-token engine envelope. Use a persistent Triton
GDN engine per backbone and score the 9B base and both adapters in that engine. Reference
kernels and serving/compiler caches use stable paths on the new EU network
volume; the US cache cannot be mounted across regions. Retain this capacity
constraint explicitly, and reuse populated EU caches on subsequent compatible
runs. Outputs: `results/augmented_judge_evaluation/`; logs:
`logs/runpod/augmented_judge_evaluation/`.

EU startup encountered unusually slow runtime-file reads on the network volume.
The first complete locked install remained intact, but its hardware-probe import
and an attempted local copy were stopped after observing FUSE waits and low
throughput. Preserve these setup logs separately from numerical gates. Recovery
installs the identical `uv.lock` into `/tmp/gleipnir-eval-fast`, with a local
package download cache because of the diagnosed volume I/O problem. The original
network environment/cache remain preserved. This changes file placement, not
dependency versions or evaluation numerics. Model weights, FLA/kernel and
compiler/serving caches remain under the stable network-volume `.cache/` paths.
Repeat the bounded hardware/runtime probe and adapter-specific serving checks
using the recovered runtime before population scoring.

## Completed result

All four fresh conditions completed: augmented 4B and base/regular/augmented 9B,
16,752 predictions in total. The 4,188 cached regular-4B predictions are reused.
All five bounded base/adapter parity cells passed before population scoring.
Independent collection checks match 27 remote artifacts/logs, 37 frozen paths,
all input IDs/labels/prompts, finite raw-logprob reconstruction and zero
truncation. Local reporting exactly reproduces the remote summary.

Preferred-answer-injected accuracy is 58.36% / 62.65% for regular / augmented
4B and 65.53% / 64.33% / 72.37% for base / regular / augmented 9B. Augmented
clean accuracies remain 99.40% / 99.60%. Preferred-injection correct-to-wrong
flips remain 37.08% / 27.63%; every augmented query loses mean p(correct)
relative to its clean counterpart. Improvement is partial and uneven, with
the hardest MT-Bench query at only 23.44% / 34.82% augmented accuracy.
The [finding](../../docs/findings/augmented_judge_evaluation.md) records paired
continuous scores, source/query breakdowns, ranking, calibration, ties and limits.
No model promotion follows.

Evaluation processes have exited and final GPU health is clean. The EU pod
remains running at $6.79/hour; both persistent volumes remain preserved.
Main-branch implementation commit: `d6f057e`. Audit/report artifacts live under
`results/augmented_judge_evaluation/`, including `collection_audit.json`,
`summary.json`, `remote_summary.json` and `preferred_injection_breakdown.json`.
