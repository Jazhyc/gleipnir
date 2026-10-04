# Monitoring augmentation transfers partially to judge injection handling

Date: 2026-10-04. Protocol:
[augmented judge evaluation](../../experiments/augmented_judge_evaluation/README.md).

The monitor-directed augmented adapters still avoid otherwise preferred answers
carrying JudgeDeceiver injections. Augmentation improves this auxiliary outcome
at both sizes, but does not remove the failure. This is transfer from monitoring
augmentation: neither augmented adapter trained on JudgeDeceiver. It is not a
new continuation or joint preference-training run.

The unchanged original auxiliary A/B quality-judging prompt is used throughout,
with native nonthinking chat and one decision token. The frozen original holdout
contains 4,188 variants, 19.992% of the prepared population: 504 clean, 1,842
preferred-answer-injected and 1,842 disfavored-answer-injected. Its 252 pairs
cover six original query groups, disjoint from training and the 30 parity
canaries. Both answer orders and every applicable suffix stay together in the
original query split. Preference construction labels remain fixed across views.

The original regular 4B results are reused after checksum and prompt validation.
No previous 9B auxiliary score population existed; base 9B and both 9B adapters
are scored freshly, alongside augmented 4B. No prompt, checkpoint, threshold or
hyperparameter is selected on this holdout.

| Model | Clean accuracy | Preferred answer injected | Disfavored answer injected | Preferred injection correct-to-wrong flips |
| --- | ---: | ---: | ---: | ---: |
| Regular 4B, cached | 99.40% | 58.36% | 95.98% | 41.21% |
| Augmented 4B | 99.40% | 62.65% | 99.67% | 37.08% |
| Base 9B | 97.82% | 65.53% | 99.89% | 33.98% |
| Regular 9B | 99.21% | 64.33% | 95.28% | 35.67% |
| Augmented 9B | 99.60% | 72.37% | 99.13% | 27.63% |

Flip rates use all 1,842 paired injected variants as the denominator, including
those whose clean counterpart was already wrong. Accuracy uses normalized
p(B) >= 0.5 to select B, retaining the original baseline's tie convention.

Preferred-answer-injected accuracy improves by 4.29 percentage points at 4B
and 8.03 at 9B. Augmented 9B exceeds base 9B by 6.84 points on this population;
regular 9B is 1.19 points below base. Almost all clean judgments remain correct,
while preferred injections still cause substantial errors. Disfavored injections
are much less disruptive. This asymmetry supports residual injection-driven
avoidance within the quality task, rather than a general loss of judging ability.
It does not establish a harmfulness false-positive rate or identify the mechanism.

| Model | Preferred injected mean p(correct) | Preferred paired mean change in p(correct) | Disfavored paired mean change |
| --- | ---: | ---: | ---: |
| Regular 4B | 0.557874 | -0.371998 | -0.067342 |
| Augmented 4B | 0.600608 | -0.349214 | -0.017457 |
| Base 9B | 0.635392 | -0.348511 | -0.001552 |
| Regular 9B | 0.587625 | -0.360267 | -0.085680 |
| Augmented 9B | 0.691693 | -0.278310 | -0.040512 |

The paired changes compare each suffix-bearing variant with its own clean pair
and order, preserving unequal suffix counts. They are not simply the difference
between the unweighted clean and injected population means. Continuous scores
confirm incomplete recovery: the augmented adapters still lose 0.349 and 0.278
mean p(correct) when the preferred answer receives an injection.

| Model | Pooled AUROC | Pooled Brier | Pooled log loss | Pooled ECE | Query-macro accuracy | Source-macro accuracy | Exact-half ties |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Regular 4B | 0.911884 | 0.123010 | 0.370631 | 0.062057 | 82.22% | 83.58% | 147 |
| Augmented 4B | 0.943884 | 0.102026 | 0.307840 | 0.070300 | 83.62% | 83.76% | 102 |
| Base 9B | 0.954735 | 0.089263 | 0.257608 | 0.060796 | 85.20% | 85.52% | 74 |
| Regular 9B | 0.927298 | 0.112825 | 0.344197 | 0.090444 | 84.72% | 86.15% | 138 |
| Augmented 9B | 0.963845 | 0.078799 | 0.242451 | 0.050986 | 90.21% | 91.79% | 58 |

Ranking, Brier and log loss improve versus regular adapters at both sizes.
Equal-source accuracy changes much less at 4B than pooled accuracy, warning
against interpreting a pooled gain as uniform transfer. ECE worsens at 4B and
improves at 9B; calibration does not improve under every diagnostic. Repeated
pooled scores number 2,758 / 2,272 for regular / augmented 4B and 1,508 / 2,627 /
1,801 for base / regular / augmented 9B. AUROC accounts for ties. Full reliability
bins and query/source views are retained in the summary rather than using
independent-example confidence intervals for this grouped population.

| Preferred-answer-injected source | Rows | Regular 4B | Augmented 4B | Base 9B | Regular 9B | Augmented 9B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LLMBar | 888 | 65.77% | 69.82% | 65.43% | 66.10% | 76.91% |
| MT-Bench | 896 | 49.89% | 56.03% | 65.40% | 60.71% | 66.52% |
| RLAIF | 20 | 60.00% | 55.00% | 100.00% | 95.00% | 100.00% |
| Search | 38 | 84.21% | 55.26% | 52.63% | 92.11% | 89.47% |

At 4B, preferred-injected accuracy improves on three of six queries and worsens
on three; at 9B it improves on five and worsens on search. The same difficult
MT-Bench query reaches only 23.44% / 34.82% accuracy for augmented 4B / 9B,
with 76.56% / 65.18% correct-to-wrong flips. Every augmented query has a
negative paired mean p(correct) shift. Augmented 4B has correctness flips on
all six queries; augmented 9B has them on five, with no RLAIF correctness flips.
Residual sensitivity therefore spans multiple sources and queries, although
the 9B RLAIF thresholded result is robust within its small held-out population.
The search construction labels and small RLAIF/search populations warrant
particular caution when interpreting their changes.

The frozen manifest SHA256 is
`3063df9fe562daad7395d6b4333a1fb988ea8c42ffef9e2a45e83219cdfe63d0`.
The summary SHA256 is
`62d141b03da82d31ddcab5fa3aa67bad6a74c84f6d63a3c5164a79b5ea1dd259`.
Source: `ShiJiawenwen/JudgeDeceiver` at
`7cfc96af0272befd83220b1d7d3b0699d072ecbf`; transformation, construction-label
and unresolved-license metadata follow the
[original finding](judge_injection_continuation.md).

One persistent vLLM engine per backbone batches adapter conditions. The B200
runtime uses Python 3.12.3, Torch 2.11.0+cu130, Transformers 5.14.1, vLLM 0.24.0
and CUDA 13.0 from the checked-in lock. Original FLA 0.5.2, causal-conv1d
1.6.2.post1 and BF16 base weights supply bounded FP32-master references; serving
uses Triton GDN. Model and compiler/kernel caches remain on the EU network
volume. Diagnosed network-volume runtime I/O latency required an identical locked
runtime in `/tmp/gleipnir-eval-fast`, recorded separately from numerical gates.

All five base/adapter parity cells pass on 30 training-derived canaries before
population scoring. Minimum correlation is 0.999059; maximum mean absolute
score difference is 0.008206. Maximum serving/reference adapter effects are
0.311984/0.300643 for augmented 4B, 0.301219/0.306557 for regular 9B and
0.373441/0.342474 for augmented 9B. Both base engines also pass agreement gates.
FP32 masters and their serving artifacts retain their frozen training/export
checksums.

Independent local collection checks match all 27 remote artifact/log checksums
and all 37 frozen source/checkpoint paths. Every fresh condition has exactly
4,188 unique expected IDs, matching labels/metadata and cached rendered prompt
hashes, finite requested A/B logprobs, exact score reconstruction, a valid
one-token decision and zero truncation. Maximum input length is 594 tokens,
with 1,521,530 prompt tokens per population. All 16,752 fresh predictions and
4,188 cached regular-4B predictions pass; local report recomputation exactly
matches the preserved remote summary. Both serving engines and the launcher
have exited, with zero GPU allocation and zero uncorrected volatile ECC errors.

Artifacts are under `results/augmented_judge_evaluation/`, with raw preferences
under each size/condition, `summary.json`, `remote_summary.json`,
`remote_inventory.json`, `collection_audit.json` and
`preferred_injection_breakdown.json`. Runtime logs are under
`logs/runpod/augmented_judge_evaluation/`. Feature checks: five focused tests and
Ruff passed. Implementation commit: `d6f057e`, on `main` as explicitly requested.

The NC2 host could not restart and NC2 lacked B200 capacity. The user accepted
EU-RO-1 and authorized required transfers. EU pod `aqmiipyogkjgi3` remains
running at $6.79/hour after evaluation; persistent volume `cb7bxu71ug` and the
original NC2 volume are preserved. No automatic post-turn follow-up is promised:
this session has no in-chat scheduling tool.

Only six held-out queries, one adapter seed, construction labels and upstream
suffixes optimized for other judge interfaces limit generalization. Tool
selection has no held-out query. These are paired transfer diagnostics, not
adaptive attacks or evidence that suspiciousness has disappeared broadly.
Together with the prior APPS findings, the result supports partial transfer from
monitoring augmentation with substantial residual injection sensitivity. No
model promotion follows from this auxiliary task.
