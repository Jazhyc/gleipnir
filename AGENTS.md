# Repository Guidelines

## Mission

Gleipnir develops a monitoring foundation model for AI control. The intended
scope includes deception, misaligned or strategically harmful actions, and other
control-relevant behavior. Qwen 3.5 is the default initial backbone, not a
permanent architectural constraint. Optimize for generalization across tasks,
model families, and deployment settings rather than benchmark-specific tricks.

## Before You Work

Read `README.md`, `docs/research_program.md`, and the README in the experiment
directory you will touch. Read relevant finding and decision records under
`docs/` before changing data contracts, prompts, objectives, or evaluation.
Update the docs when an experiment changes what the project should believe.

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
condition before launching an expensive run. Prefer continuous scores and report
ranking metrics, calibration, threshold diagnostics, score ties, and performance
by task/source/model family. Use grouped holdouts whenever examples share a source
conversation, generator, or annotation lineage. Never promote on the final test
set or hide negative results.

Teacher caches must be resumable and prompt-aware. Record model/provider IDs,
request settings, prompt hashes, raw returned logprobs, normalized targets, token
usage, timestamps, and parse failures. Student training must distinguish hard
labels, privileged rationale targets, and soft teacher distributions. Preserve
FP32 master adapters; export lower-precision inference copies only after matched
parity checks.

## Compute

Use local Slurm GPU jobs for cluster experiments. Default to one `gpushort`
`rtx_pro_6000` GPU, one CPU, and 32 GB RAM unless the workload requires a
documented change. Redirect final logs to `logs/slurm/<experiment>/` and remove
the temporary bootstrap output after redirection. Batch related inference
conditions in one persistent vLLM process when possible.

Run Slurm control and submission commands outside the filesystem/network
sandbox. Restricted shells can block name resolution or controller sockets and
make a healthy `slurm1` appear down. Before diagnosing a controller outage,
repeat `scontrol ping` with sandbox escalation. Use ordinary `sbatch`/`salloc`
jobs to hold GPUs: advanced reservations created with `scontrol` require Slurm
administrator privileges.

The active reserved Lambda Cloud training target is `gleipnir-improvement`. Use `scripts/lambda_cloud.py` for SSH,
sync, bootstrap, secret transfer, and artifact collection. Probe and record the
active target's hardware before freezing a recipe. Never terminate it or launch
billable capacity without the user's explicit instruction. Pull important
artifacts before any termination.

For text-only Qwen3.5 training on H100s, do not silently use Transformers'
Torch gated-delta fallback. Use the causal-LM loader and verify the isolated,
pinned `flash-linear-attention==0.5.2` and `fla-core==0.5.2` kernels before model
import. The proven eager recipe is microbatch 8 with gradient accumulation 4
(effective batch 32) and `torch.compile=false`; preserve that recipe unless a
matched benchmark supports a change. Prefer standard QLoRA (4-bit NF4, double
quantization, BF16 compute) when it permits the proven batch at high adapter
ranks. Preflight the largest rank, fail closed if FLA is unavailable, and record
kernel, quantization, batch, memory, and throughput metadata for every campaign.
For direct-boundary objectives, avoid materializing full-sequence vocabulary
logits when it prevents the proven batch; use a matched, recorded selected-token
projection consistently across the campaign.

For frozen text-only evaluation of standard base or PEFT LoRA models, default to
one persistent vLLM engine with continuous batching, a constrained one-token
response, and explicitly requested decision-token logprobs. Do not carry an
eager training microbatch into large evaluation runs. Use Transformers eager
evaluation only for a bounded backend-parity canary or when vLLM cannot represent
the model/adapter, and document that exception. For every new adapter layout or
backend combination, compare the master checkpoint with its serving artifact and
record score agreement plus a nonzero adapter effect before scaling evaluation.

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
