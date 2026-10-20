# Hosted decision models on the strict-OOD suite

Evaluation: 2026-10-20. Status: all 6,395 frozen OOD IDs scored exactly once
per model, with no input rejections or fallbacks. The pre-registration,
config and runner are in
[`experiments/decision_models_ood`](../../experiments/decision_models_ood/README.md).

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
