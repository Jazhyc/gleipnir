# Decision models on the strict-OOD suite

Benchmarks four hosted decision models through OpenRouter's Decisions API:

- Microsoft-Decision-1 (Azure)
- Mercury Decide (Inception)
- d1-ultra (Liquid)
- d1 (Liquid), the weaker model in Liquid's family

## Pre-registration (frozen 2026-10-20, before any paid call)

- **Hypothesis.** Native decision models score the frozen 6,395-row strict-OOD
  suite at hosted costs comparable to Jev 1.13 and pplx-decider v1.2. Any of
  them may add a frontier point.
- **Intervention.** Identical to the Jev 1.13 benchmark:
  - **State:** the raw trajectory (frozen Jev states, same checksums).
  - **Question:** the frozen Jev Noul question (`prompts/jev_noul_v1.json`,
    question SHA-256 `9b423177…`).
  - **Score:** the returned probability.
  - **Routing:** each request is pinned to the model's provider, with no
    fallbacks and data collection denied.
- **Baselines.** Jev 1.13 (0.8495 at $0.3887/1K) and pplx-decider v1.2
  (0.8956 at $0.1763/1K), on the same rows.
- **Phases.**
  - **canary:** the 100-row non-OOD Jev canary. This is an interface gate,
    not a quality gate, so weaker models are still evaluated. It requires all
    rows answered, the pinned provider and revision, and spend ≤ $0.10.
  - **longest:** the longest OOD state, label-blind. Its score is withheld
    and cannot change the campaign.
  - **full:** all 6,395 rows, exactly once.
- **Selection.** Nothing is tuned on OOD; each model has one frozen
  configuration.
- **Rejected inputs.** States the provider rejects as input (HTTP
  400/413/422, such as over the 32K windows of Microsoft-Decision-1 and
  d1-ultra) are recorded. They score 0.5 in the primary metric, and
  answered-only metrics are reported beside it.
- **Cost.**
  - **Reporting:** `input tokens × OpenRouter list price` at
    `config.json` prices. Output is free. Mercury Decide's temporary 50%
    discount is not applied.
  - **Ceilings:** $3.25 per full run at $0.042/M and $1.55 at $0.02/M.
- **Stop conditions.** Any of these stops a model's run:
  - a failed canary gate;
  - any non-input error after retries (auth, credits, routing);
  - a different provider in a response;
  - input rejections above 2% of rows;
  - reported spend above the ceiling;
  - question, input or request-settings hash drift.

### Amendment: Gemini Decision 1 (2026-11-18, before any paid call)

The user added Gemini Decision 1 (`google/gemini-decision-1`, pinned to
Google AI Studio, revision `-20261111`, 262,144-token window, $0.10/M input).
Every other part of the protocol is unchanged. At $0.10/M its 100-row canary
costs about $0.11, above the shared $0.10 canary ceiling, so the config gives
it a $0.15 canary ceiling and a $7.50 full-run ceiling (about 59M tokens
expected, $5.90).

### Amendment: pplx-decider v1.3 (2026-11-18, before the canary)

The user added pplx-decider v1.3 through its free OpenRouter endpoint
(`perplexity/pplx-decider-v1.3:free`, pinned to Perplexity, revision
`-20261112`, 262,144-token window). Billed spend is $0, so the spend ceilings
are $0.01. Costs are reported at the $0.02/M list price of v1.2, because a
free price would dominate every frontier point; the endpoint may stop being
free. Unlike the v1.2 row, which was scored locally from the readout logit
margin, this row uses the hosted endpoint's probability and its billed input
tokens, like the other models of this campaign. One probe request on a canary
row (2026-11-18) confirmed the endpoint accepts these settings.

### Amendment: Jev 1.14 (2026-11-19, before any paid call)

The user added Jev 1.14 (`typesafe/jev-1.14`, pinned to TypeSafe, revision
`-20261014`, 262,144-token window, $0.042/M input, free output; endpoint
listing checked 2026-11-19), so that both TypeSafe and Perplexity are
represented by their newest model. Every other part of the protocol,
including the shared ceilings, is unchanged; about 60M tokens are expected,
$2.50 at list price.

## Running

```bash
python -m experiments.decision_models_ood.run run <model> canary
python -m experiments.decision_models_ood.run run <model> longest
python -m experiments.decision_models_ood.run run <model> full
```

Model keys: `microsoft_decision_1`, `mercury_decide`, `d1_ultra`, `d1`,
`gemini_decision_1`, `pplx_decider_v1_3`, `jev_1_14`.
Outputs go to `results/decision_models_ood/<model>/<phase>/`.

## Result

All four runs completed with no rejections. None reaches the frontier:
d1-ultra 0.8929 ($0.3472/1K), Mercury Decide 0.8828 ($0.1811/1K),
Microsoft-Decision-1 0.7506 ($0.3887/1K), d1 0.6742 ($0.1653/1K).
Gemini Decision 1 (amendment, 2026-11-18) scores 0.8673 at $0.9306/1K, also
off the frontier; Google's filter blocked 2 MLE-Sabotage inputs.
pplx-decider v1.3 (amendment, 2026-11-18, free endpoint, priced at v1.2's
$0.02/M) scores 0.8942 at $0.1760/1K, just below v1.2. Jev 1.14
(amendment, 2026-11-19) scores 0.8763 at $0.3887/1K, above Jev 1.13 (0.8495)
and off the frontier. See
[the finding](../../docs/findings/decision_models_ood_benchmark.md).
