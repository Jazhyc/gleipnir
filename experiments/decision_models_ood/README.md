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

## Running

```bash
python -m experiments.decision_models_ood.run run <model> canary
python -m experiments.decision_models_ood.run run <model> longest
python -m experiments.decision_models_ood.run run <model> full
```

Model keys: `microsoft_decision_1`, `mercury_decide`, `d1_ultra`, `d1`.
Outputs go to `results/decision_models_ood/<model>/<phase>/`.

## Result

All four runs completed with no rejections. None reaches the frontier:
d1-ultra 0.8929 ($0.3472/1K), Mercury Decide 0.8828 ($0.1811/1K),
Microsoft-Decision-1 0.7506 ($0.3887/1K), d1 0.6742 ($0.1653/1K). See
[the finding](../../docs/findings/decision_models_ood_benchmark.md).
