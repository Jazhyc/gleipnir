# Hosted decision models on the strict-OOD suite

Evaluation: 2026-10-20. Status: all 6,395 frozen OOD IDs scored exactly once
per model, with no input rejections or fallbacks. The pre-registration,
config and runner are in
[`experiments/evaluation/decision_models_ood`](../../../experiments/evaluation/decision_models_ood/README.md).

## Setup

- **Interface.** Four models through OpenRouter's Decisions API, identical
  to the Jev 1.13 benchmark: the raw trajectory as state, the frozen Jev Noul
  question (SHA-256 `9b423177…`) and the returned probability as the score.
- **Inputs.** The same frozen Jev states (`c38c6214…`); the canary input is
  `7ce0840f…`.
- **Routing.** Each request is pinned to its provider, with fallbacks off
  and data collection denied. Every response came from the pinned provider
  and dated revision.
- **Cost.** Billed input tokens × OpenRouter list price; output is free.
  Reported spend equalled list-price cost for every model. Mercury Decide's
  temporary 50% discount was not applied in billing and is not used here.

## Results

| Model (revision) | Mean-OOD pAUROC@20 | Mean-OOD AUROC | Macro Brier | Input tokens | $/M | $/1K |
|---|---:|---:|---:|---:|---:|---:|
| d1-ultra (`liquid/d1-ultra-20261015`) | **0.892904** | 0.952350 | 0.0954 | 52,861,752 | 0.042 | 0.347176 |
| Mercury Decide (`inception/mercury-decide-20260930`) | 0.882811 | 0.943141 | 0.0842 | 57,899,352 | 0.02 | 0.181077 |
| Microsoft-Decision-1 (`microsoft/microsoft-decision-1-20261009`) | 0.750638 | 0.905627 | 0.1786 | 59,178,862 | 0.042 | 0.388665 |
| d1 (`liquid/d1-20260930`) | 0.674230 | 0.869197 | 0.2888 | 52,861,752 | 0.02 | 0.165322 |

| Per-source pAUROC@20 | AgentDojo | CUA-SHADE | IaC | MLE | Rogue | SHADE-Arena |
|---|---:|---:|---:|---:|---:|---:|
| d1-ultra | 0.9939 | 0.8240 | 0.9901 | 1.0000 | 0.9931 | 0.5563 |
| Mercury Decide | 0.9487 | 0.8359 | 0.9926 | 0.9949 | 0.9768 | 0.5479 |
| Microsoft-Decision-1 | 0.8733 | 0.6692 | 0.6722 | 0.9735 | 0.7655 | 0.5502 |
| d1 | 0.7706 | 0.5755 | 0.8390 | 0.9612 | 0.4729 | 0.4263 |

- **Frontier.** No model joins either frontier.
  - pplx-decider v1.2 (0.8956 at $0.1763/1K) is cheaper and better than
    both Mercury Decide and d1-ultra.
  - Jev 1.13 (0.8495 at $0.388664/1K) is $0.000001/1K cheaper than
    Microsoft-Decision-1 and 10 points better.
  - Gleipnir 4B (0.7703 at $0.0850/1K) dominates d1.
- **SHADE-Arena.** It is the weakest source for every model (0.43–0.56),
  as it is for Jev (0.44) and the Decider (0.61).
- **Tokenizers.**
  - Microsoft-Decision-1 billed 59,178,862 input tokens, 110 more than Jev
    on the same states (59,178,752).
  - d1 and d1-ultra billed identical counts, so they share a tokenizer.
- **Score resolution.** Unique scores: Microsoft-Decision-1 6,385, Mercury
  Decide 6,039, d1 4,161, d1-ultra 3,747. All are far finer than Jev's 95.
- **Context.** The longest billed inputs were 32,890 (Microsoft-Decision-1,
  listed window 32,768), 32,069 (Mercury), and 29,678 (both d1 models).
  Microsoft-Decision-1 accepted one state slightly over its listed window
  without error; truncation cannot be ruled out for that row.
- **Canary** (100 non-OOD rows, interface gate only). pAUROC@20 was
  Mercury 0.940, d1-ultra 0.932, Microsoft 0.776, d1 0.696, which ranks the
  models the same way as OOD.

## Spend

Total about $7.06: full runs $6.92 (Microsoft $2.4855, d1-ultra $2.2202,
Mercury $1.1580, d1 $1.0572), canaries $0.14, longest-row checks $0.004.

## Artifacts

`results/decision_models_ood/<model>/{canary,longest,full}/`, each with
`scores.jsonl` (raw responses, usage, provider, revision) and `summary.json`.
Full-run score SHA-256 prefixes: Microsoft `0343ec5cf60d`, Mercury
`92d78632fcea`, d1-ultra `8504e1dcb8fd`, d1 `cb7f7b63b0aa`.

## Gemini Decision 1 (2026-11-18)

Added by amendment ([pre-registration](../../../experiments/evaluation/decision_models_ood/README.md)),
same protocol, pinned to Google AI Studio (`google/gemini-decision-1-20261111`,
262,144-token window, $0.10/M input).

| Model | Mean-OOD pAUROC@20 | Mean-OOD AUROC | Macro Brier | Input tokens | $/M | $/1K |
|---|---:|---:|---:|---:|---:|---:|
| Gemini Decision 1 | 0.867342 | 0.949072 | 0.1825 | 59,509,013 | 0.10 | 0.930555 |

| Per-source pAUROC@20 | AgentDojo | CUA-SHADE | IaC | MLE | Rogue | SHADE-Arena |
|---|---:|---:|---:|---:|---:|---:|
| Gemini Decision 1 | 0.9785 | 0.8173 | 0.8539 | 0.9556 | 0.9748 | 0.6239 |

- **Frontier.** It is off both frontiers: pplx-decider v1.2 (0.8956 at
  $0.1763/1K), Mercury Decide and d1-ultra are each cheaper and better.
  SHADE-Arena (0.624) is its weakest source but the best among the hosted
  decision models.
- **Refusals.** Google's input filter blocked 2 rows (HTTP 400, "sensitive
  words that violate Google's Generative AI Prohibited Use policy"), both
  MLE-Sabotage (one honest, one attack). They score 0.5 under the protocol;
  over the 6,393 answered rows Mean-OOD pAUROC@20 is 0.8742. No other
  decision model refused an input.
- **Scores.** Probabilities lie in [0.01, 0.99] with 4,888 unique values;
  the canary's 69 unique values on 100 rows did not reflect coarse
  resolution.
- **Canary** (interface gate): pAUROC@20 0.893, below Mercury (0.940) and
  d1-ultra (0.932), matching its OOD rank.
- **Spend.** Full run $5.9509, canary $0.1161, longest-row check below
  $0.01. Full-run score SHA-256 prefix `41210ee24721`.

## pplx-decider v1.3 (2026-11-18)

Added by amendment, same protocol, through the free OpenRouter endpoint
(`perplexity/pplx-decider-v1.3:free`, pinned to Perplexity, revision
`-20261112`). Billed spend was $0; the cost uses the $0.02/M list price of
v1.2 (user decision), because a free price would dominate every frontier
point.

| Model | Mean-OOD pAUROC@20 | Mean-OOD AUROC | Macro Brier | Input tokens | $/M | $/1K |
|---|---:|---:|---:|---:|---:|---:|
| pplx-decider v1.3 | 0.894199 | 0.950634 | 0.0946 | 56,280,923 | 0.02 (v1.2) | 0.176015 |

| Per-source pAUROC@20 | AgentDojo | CUA-SHADE | IaC | MLE | Rogue | SHADE-Arena |
|---|---:|---:|---:|---:|---:|---:|
| pplx-decider v1.3 (hosted probability) | 0.9972 | 0.8427 | 0.9953 | 0.9980 | 0.9785 | 0.5535 |
| pplx-decider v1.2 (local margin) | 0.9932 | 0.8537 | 0.9967 | 0.9878 | 0.9367 | 0.6054 |

- **Against v1.2.** 0.0014 lower than v1.2 (0.8956 at $0.1763/1K): better
  on Rogue Eval (+0.042) and MLE, worse on SHADE-Arena (−0.052) and
  CUA-SHADE-Arena. The two rows differ in readout: v1.2 was scored locally
  from the readout logit margin, v1.3 from the hosted probability, which
  had 6,358 unique values over 6,395 rows, so ties are not the cause.
- **Frontier.** Off both frontiers: it is $0.0002/1K cheaper than v1.2 but
  less accurate, and the 2% cost tolerance of the frontier rule leaves v1.2
  on the frontier. Under the strict rule it would be a frontier point.
- **Tokens.** The endpoint billed 56,280,923 input tokens, 78,576 fewer than
  the local v1.2 joint-sequence count (56,359,499).
- **Canary** (interface gate): pAUROC@20 0.816, against 0.844 for v1.2 on
  the same rows; neither predicted the OOD score well.
- **Artifacts.** `results/decision_models_ood/pplx_decider_v1_3/`; full-run
  score SHA-256 prefix `db97ecfe63b9`.

## Jev 1.14 (2026-11-19)

User-requested so that TypeSafe's newest model is measured as Perplexity's
v1.3 was ([amendment](../../../experiments/evaluation/decision_models_ood/README.md#amendment-jev-114-2026-11-19-before-any-paid-call)).
`typesafe/jev-1.14`, pinned to TypeSafe, revision `jev-1.14-20261014`. The
canary passed ($0.049), and all 6,395 OOD rows answered on the first attempt
with no rejections: 59,178,752 input tokens, $2.49 at $0.042/M.

| Monitor | Mean-OOD pAUROC@20 | Mean AUROC | Recall at 1% FPR | USD / 1K |
|---|---:|---:|---:|---:|
| Jev 1.14 | 0.8763 | 0.9393 | 0.783 | 0.3887 |
| Jev 1.13 | 0.8495 | 0.9269 | 0.733 | 0.3887 |

| Benchmark | AgentDojo | CUA-SHADE | IaC | MLE | Rogue Eval | SHADE |
|---|---:|---:|---:|---:|---:|---:|
| Jev 1.14 pAUROC@20 | 0.9959 | 0.8340 | 0.9808 | 0.9934 | 0.9341 | 0.5194 |

- **Better than 1.13 at the same price**, by 0.027 pAUROC@20, but off the
  frontier: pplx-decider v1.2 (0.8956) and Mercury Decide (0.8828) score
  higher at under half the cost.
- **Coarse scores.** 97 distinct values over 6,395 rows (0.03 to 0.99), as
  for 1.13.
- On SleightBench (stripped view, ControlArena prompt, run `sb16`) it scores
  0.260 pAUROC@20 and catches 20% at 1% FPR
  ([finding](sleightbench_released_monitors.md)).
- In the report figures Jev 1.14 replaces Jev 1.13 (user, 2026-11-19); the
  tables keep both.
- **Artifacts.** `results/decision_models_ood/jev_1_14/`.

## Drex Max (2026-11-20)

User-requested after the Drex report (arXiv:2611.07854) was cited in the
technical report ([amendment](../../../experiments/evaluation/decision_models_ood/README.md#amendment-drex-max-2026-11-20-before-any-paid-call)).
`nace-ai/drex-max`, pinned to DeepInfra, revision `drex-max-20261016`. The
canary passed ($0.044), and all 6,395 OOD rows answered with no rejections:
37,974,756 billed input tokens, $2.28 billed at the discounted $0.06/M,
**$4.56 at the undiscounted $0.12/M** used for the cost (user, 2026-11-20).

| Monitor | Mean-OOD pAUROC@20 | Mean AUROC | Recall at 1% FPR | USD / 1K |
|---|---:|---:|---:|---:|
| Drex Max | 0.8299 | 0.9334 | 0.566 | 0.7126 |
| Jev 1.14 | 0.8763 | 0.9393 | 0.783 | 0.3887 |
| pplx-decider v1.3 | 0.8942 | 0.9506 | 0.815 | 0.1760 |

| Benchmark | AgentDojo | CUA-SHADE | IaC | MLE | Rogue Eval | SHADE |
|---|---:|---:|---:|---:|---:|---:|
| Drex Max pAUROC@20 | 0.7924 | 0.8420 | 0.9721 | 0.9918 | 0.9413 | 0.4396 |

- **Below Jev 1.14 and Decider v1.3, off the frontier.** Its lowest scores
  are SHADE-Arena (0.440) and AgentDojo (0.792), where the other decision
  models score near 1 on AgentDojo.
- **Billing rule.** DeepInfra's endpoint bills long states at fewer tokens
  than their Qwen3.8 length: in full below about 10K tokens, about 10.3K up
  to about 33K, then 0.314 of the length. No OOD row was billed above 10,637
  tokens, against up to about 28K for the other models. Probes on
  2026-11-20 (needle at start, middle or end of filler and of random text,
  up to 140K tokens) show the model reads the whole state, and Drex Max's
  published tokenizer gives the same counts as Qwen3.8-27B's, so this is a
  billing rule, not truncation or a different tokenizer. Costs use the
  billed tokens, as for every API model.
- **Scores.** 484 distinct values over 6,395 rows, 0.0009 to 0.9997.
- **Retries.** 390 rows needed more than one attempt (at most six), all
  within the protocol's retry budget.
- On SleightBench (stripped view, ControlArena prompt, run `sb17`): 0.322
  pAUROC@20, 25% catch at 1% FPR ([finding](sleightbench_released_monitors.md)).
- In the report, Drex Max appears in the OOD table but not in the OOD
  figures (off the frontier, like Gemini Decision 1), and in the SleightBench
  figures and table.
- **Artifacts.** `results/decision_models_ood/drex_max/`.

