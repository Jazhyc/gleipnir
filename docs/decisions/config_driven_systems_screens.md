# Config-driven monitoring systems screens

Status: adopted 2026-09-04.

## Decision

New matched training-throughput ablations use
`gleipnir.monitoring_systems_screen` instead of creating separate `core.py`,
`prepare.py`, `run_lambda.py`, and `summarize.py` implementations. Each
hypothesis normally needs only:

- a README stating the question, interpretation, and exact invocation; and
- one versioned JSON contract containing data identities, the common recipe,
  condition overrides, metadata expectations, preflight, and promotion rule.

The shared runner owns deterministic stratified and longest-row selections,
input checksums, exact job reconstruction, pinned-kernel checks, compile and
checkpoint metadata validation, same-weight canaries, isolated compiler caches,
serial per-GPU lanes, atomic status, and throughput summaries.

```bash
python -m gleipnir.monitoring_systems_screen prepare --config CONFIG.json
GLEIPNIR_COMMIT=$(git rev-parse HEAD) \
  python -m gleipnir.monitoring_systems_screen run --config CONFIG.json
```

The inner-mixer compilation experiment is the first migrated contract. Its
historical artifacts reproduce the same selection hashes and summary under the
shared implementation.

## Boundaries

The README and frozen contract remain separate because they carry scientific
intent and auditable decisions rather than boilerplate. A materially new data
contract, objective, evaluation, or multi-stage algorithm can still justify
custom code. The shared runner is for matched systems screens where conditions
change explicit training-job fields and selection is based on measured
throughput.

Completed historical experiments are not rewritten merely to reduce file
count. They can be migrated when reused; new screens should start from the
shared runner.
