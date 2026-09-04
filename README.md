# Gleipnir

Gleipnir is a research project for building a monitoring foundation model for AI
control. The aim is to distill broad, calibrated judgments about deception,
misaligned actions, policy-relevant behavior, and other control-relevant events
into deployable monitors. Initial work uses the Qwen 3.5 family as the student
backbone and grows out of the successful Aletheia's Quest distillation line.

The repository is intentionally experiment-centric: each hypothesis gets its own
directory under `experiments/`, while reusable code lives in `src/gleipnir/` and
durable conclusions live in `docs/`.

## Released models

- [Gleipnir 4B](https://huggingface.co/Jazhyc/Gleipnir-4B)
- [Gleipnir 9B](https://huggingface.co/Jazhyc/Gleipnir-9B)

Both are MIT-licensed rank-128 LoRA research artifacts for reproducibility and
follow-up work. They score visible AI-agent trajectories for deception,
scheming, and other control-relevant problematic behavior. They are not
standalone models or production safety systems; see the model cards for the
frozen prompt, direct binary-logit interface, results, and limitations.

## Quick start

```bash
cd /scratch/s4626451/gleipnir
cp .env.example .env
./setup_dev.sh
source .venv/bin/activate
pytest
```

The lock file pins the environment. At bootstrap, the current top-level inference
stack is vLLM 0.24.0 and Transformers 5.14.1; neither NNsight nor the old
competition runner is included.

## Layout

- `src/gleipnir/`: shared prompt, API, metric, and training utilities.
- `experiments/<hypothesis>/`: one self-contained hypothesis and its launchers.
- `cluster/slurm/`: reusable Slurm entrypoints for Hábrók/RUG.
- `scripts/`: operational and plotting entrypoints, including Lambda Cloud management.
- `figures/`: tracked, reproducible figures and their regeneration commands.
- `docs/`: research program, findings, decisions, and infrastructure notes.
- `data/`, `results/`, `logs/`: ignored local artifacts; only `.gitkeep` files are tracked.

Reusable plotting conventions live in `src/gleipnir/plotting.py`. See
[`figures/README.md`](figures/README.md) for the figure registry and exact
regeneration commands.

Matched monitoring throughput ablations use the config-driven
`gleipnir.monitoring_systems_screen` runner. New systems screens normally need
only a Hydra YAML config and experiment README. Preparation resolves defaults
and overrides into the hashed JSON execution contract; see
[`docs/decisions/config_driven_systems_screens.md`](docs/decisions/config_driven_systems_screens.md).

Start with [the research program](docs/research_program.md), then read the README
inside the experiment you are changing.
