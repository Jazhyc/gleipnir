# Test map

The default `pytest` suite lives here, grouped by responsibility. Experiment-local
tests stay beside their experiments and run when that directory is selected
explicitly.

| Directory | What it checks |
| --- | --- |
| `adapters/` | Adapter merging and release packaging |
| `analysis/` | Plots, profiler summaries and audit reports |
| `campaigns/` | Training screens, scaling campaigns, launch plans and receipts |
| `data/` | Input preparation, sampling and provenance contracts |
| `evaluation/` | Metrics, calibration, scoring interfaces and evaluation runners |
| `infrastructure/` | Cloud tooling, Slurm commands and network proxy logic |
| `integration/` | Distributed training canary and its launcher/cache contracts |
| `kernels/` | Quantization layouts, reference arithmetic and benchmark contracts |
| `layout/` | Source organization, legacy imports and source identity |
| `serving/` | Inference configuration, model wiring and mocked serving behavior |
| `teachers/` | Prompts, teacher clients, annotation caches and exports |
| `training/` | Losses, batching, packing, backends and Trainer behavior |
| `helpers/` | Shared test configuration and repository paths |

Run a group with `pytest tests/training` or `pytest tests/teachers`. These checks
use synthetic inputs and mock paid APIs and remote lifecycle operations. Kernel
and serving groups check CPU references and contracts; native hardware checks
have experiment-specific runtime requirements documented in their READMEs.

The `integration` marker selects the actual two-process Gloo canary, which needs
local networking. Its companion contract tests are ordinary CPU checks:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/pytest -m "not integration"
.venv/bin/pytest -m integration
.venv/bin/pytest experiments/b200_bf16_fa4
```

Plain `pytest` still collects the entire suite here, including the integration
canary. Importlib collection also allows experiment directories with repeated
test filenames to be checked together.
