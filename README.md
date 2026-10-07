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
- `outputs/presentations/<presentation_name>/`: self-contained LaTeX/Beamer
  presentations, with committed public PDFs, local slides, styling, assets and
  build instructions.
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
The user-selected recipe for future single-B200 Qwen3.5-4B training is
`systems_screen@_global_: qwen35_4b_b200_default`: native NVFP4 MLP forward and
base input gradients with hardware activation packing and fused descale,
FP32 master adapters, BF16 FlashQLA, native variable-length FlashAttention 4,
no model checkpointing and selectively compiled physical rows with a
16,384-token budget and logical
batch 32. See the
[recipe decision](docs/decisions/b200_native_fp4_training_recipe.md) for the
checksum-bound startup reuse, explicit finite acceptance and numerical limits.
The BF16 comparison is `qwen35_4b_b200_bf16_fa4`; frozen campaigns retain their
recorded precision and receipt rather than switching recipes mid-campaign.
The preceding fixed-batch, checkpoint and FA4 screen remains recorded in
[`docs/findings/b200_training_throughput.md`](docs/findings/b200_training_throughput.md).

Start with [the research program](docs/research_program.md), then read the README
inside the experiment you are changing.

The B200 inference optimization baseline uses FROST FP4 MLPs, large GDN
and full-attention projection GEMMs, symbolic-row SwiGLU overhead improvements,
direct packed FP4 MLP activation output for large batches, and cuDNN MXFP8 full-attention
prefill. Recurrence, KV cache and decode remain BF16, with FP32 gates/state. The user accepts the
development quality tradeoff; strict failed parity remains recorded separately.
Native Gigatoken encoding and direct FROST host binding reuse are now the
selected serving reference, with c1 median/p95 149.69/184.19 ms and c128
throughput 197530 input tokens/s. See the
[serving reference decision](docs/decisions/b200_gigatoken_direct_host_reference.md).

A cache-free whole-prompt monitoring prototype reduces c1 latency and persistent
GPU storage, but its repeated batch run stalls. Keep it experimental and retain
the selected serving reference; see the
[cache-free finding](docs/findings/b200_cache_free_serving.md).

Presentation layout and build conventions are documented in
[`outputs/presentations/README.md`](outputs/presentations/README.md).
