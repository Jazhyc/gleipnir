# Decision log

## 2026-08-09 — Bootstrap

- Start Gleipnir as a standalone repository rather than retaining competition
  submission and leaderboard machinery.
- Use Qwen 3.5 as the initial backbone family.
- Seed the project with deception distillation, generic OpenRouter binary-logprob
  collection, Slurm tooling, and Lambda Cloud campaign management.
- Pin vLLM 0.24.0 and Transformers 5.14.1, the latest stable PyPI releases checked
  at bootstrap; exclude NNsight and NDIF.
- Keep experiments isolated and record durable findings under `docs/`.

