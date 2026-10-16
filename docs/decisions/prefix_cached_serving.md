# Opt-in prefix-cached serving

User decision, 2026-10-16: offer align-mode prefix caching as an opt-in serving
mode for synchronous monitoring, accepting its score drift. Enable it with
`experiments.b200_attention_precision.startup --prefix-caching`. It is not the
default: evaluation and benchmark runners replay identical prompts and keep
`--no-enable-prefix-caching`.

**Evidence** ([replay](../../experiments/b200_prefix_replay/README.md)):
- **Speed:** 12.9× less time to monitor every action of a 1M-token session;
  7.9× for 7.2% escalated actions.
- **Drift:** cached scores differ from uncached by 0.03–0.15 (mean 0.055),
  the same size as other measured path effects.

**Startup acceptance.** The mode's startup canary uses MAE ≤ 0.02 and
correlation ≥ 0.99, the campaigns' new-adapter serving parity limits. The
default stack keeps the 0.005 reproduction limit. The first opt-in startup
measured MAE 0.0083, correlation 0.9991. The receipt records which limits
applied.
