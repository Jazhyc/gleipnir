# Repository Guidelines

This file defines repository working rules. The
[research program](docs/research_program.md) defines project goals, scientific
direction and the current evaluation contract.

## Before You Work

Read [README.md](README.md), then the README in each experiment directory you
will touch. Load additional instructions by task:

- Research design, data contracts, prompts, objectives, evaluation methodology
  or interpretation: read [the research program](docs/research_program.md) and
  relevant findings/decisions under `docs/` before making changes. Use
  [the documentation index](docs/README.md) to locate them.
- Compute launch, monitoring, infrastructure, remote access or lifecycle work:
  read [compute instructions](docs/agent_guides/compute.md).
- Training implementation, configuration or optimization: read
  [training instructions](docs/agent_guides/training.md).
- Inference, serving, evaluation execution or kernel optimization: read
  [inference instructions](docs/agent_guides/inference.md).

Read every guide that applies when a task spans these areas. Routine code or
documentation maintenance requires only the relevant context; the research
program and historical logs are not universal startup reading. Update findings
and decisions when an experiment changes what the project should believe.
Keep changing recipes and selections in their linked decision/configuration,
and detailed run history in experiment records rather than this file.

## Project Structure

Reusable library code belongs in `src/gleipnir/`. Put each hypothesis in
`experiments/<short_name>/` with a README, configuration, entrypoint, and focused
tests. Shared Slurm launchers live in `cluster/slurm/`; remote infrastructure
tools live in `scripts/`. Store raw or materialized inputs under `data/`, outputs
under `results/<experiment>/`, and runtime logs under `logs/<platform>/<experiment>/`.
Those artifact trees are ignored by Git.

Do not copy upstream datasets into the repository. Record source identifiers,
revisions, licenses, transformations, prompt hashes, and artifact checksums so
data can be reconstructed. Treat labels and privileged teacher information as
separate fields with explicit provenance.

## Environment and Commands

Use Python 3.12 and the checked-in `uv.lock`:

```bash
./setup_dev.sh
source .venv/bin/activate
pytest
ruff check .
```

The project uses current stable vLLM and Transformers releases pinned in
`pyproject.toml`/`uv.lock`. Do not add NNsight or NDIF dependencies. Keep secrets
only in the ignored `.env` or platform secret stores; update `.env.example` with
names and descriptions, never values.

## Experiment Standards

Write the hypothesis, intervention, baselines, held-out selection rule, and stop
condition before launching an expensive run. Freeze selection and promotion
criteria before evaluating candidates. Prefer continuous scores and report
ranking metrics, calibration, threshold diagnostics, score ties, and performance
by task/source/model family. Use grouped holdouts whenever examples share a
conversation, task, source, generator or annotation lineage; keep derived views
with their parents and disclose missing lineage. Never promote on the final
test set or hide negative results.

Teacher caches must be resumable and prompt-aware. Record model/provider IDs,
request settings, prompt hashes, raw returned logprobs, normalized targets, token
usage, timestamps, and parse failures. Student training must distinguish hard
labels, privileged rationale targets, and soft teacher distributions. Preserve
FP32 master adapters; export lower-precision inference copies only after matched
parity checks.

## Compute boundaries

Never launch billable capacity or terminate an instance without the user's
explicit instruction. Collect important artifacts before termination. Preserve
persistent compiler/kernel caches, frozen campaign contracts, failed receipts
and quality artifacts. Verify live capacity before acting; historical
running-status notes are not a live inventory. Existing user authorization
continues to apply within its stated scope.

## Code, Tests, and Git

Use 4-space Python indentation, type hints for public interfaces, `snake_case`
for functions/files, and `PascalCase` for classes. Keep reusable modules small
and avoid hidden dependencies between experiment folders. Add focused tests for
non-trivial parsing, cache identity, loss functions, metrics, and launch logic.
Mock paid APIs and remote lifecycle operations in tests.

Work on a feature branch for new research methods. Keep commits scoped, use short
imperative subjects, and commit coherent completed changes. Do not commit keys,
raw paid API responses containing sensitive inputs, model weights, caches,
datasets, runtime logs, or generated result directories.

Automatically make a scoped commit whenever a complete feature is finished and
its associated run has successfully started. Run the relevant tests and inspect
the staged files first; do not wait for a separate request to commit, and never
include ignored experiment artifacts in that automatic commit.
