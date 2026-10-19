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

## Evaluation data

- **ID and OOD.** The upstream datasets declare no license, so we don't
  redistribute them. To rebuild the frozen sets locally and check them against
  their pinned SHA256s, run
  `python scripts/build_eval_datasets.py {id,ood,all}`. This needs an
  `HF_TOKEN`, Hugging Face access to the Qwen3-8B and Kimi-K3 tokenizers and
  about 5 minutes once the tokenizers are cached. It makes no paid API calls. `--verify-only` checks existing
  outputs without rebuilding.
- **APPS.** The injection benchmark and our honest-code controls are in the
  private dataset
  [`Jazhyc/gleipnir-apps-injection-evals`](https://huggingface.co/datasets/Jazhyc/gleipnir-apps-injection-evals).
  The upstream author permitted us to share it. Re-stage it with
  `python scripts/release_hf_apps_dataset.py prepare`, then push with
  `python scripts/release_hf_apps_dataset.py upload --namespace Jazhyc`.
  Uploads are private unless you pass `--public`. The card is in
  `data_releases/hf/`.

## Quick start

```bash
cd /scratch/s4626451/gleipnir
cp .env.example .env
./setup_dev.sh
source .venv/bin/activate
pytest
```

The lock file pins the environment. At bootstrap, the current top-level inference
stack is vLLM 0.31.0 and Transformers 5.14.1; neither NNsight nor the old
competition runner is included. The B200 migration and scheduler screen lives
in [its experiment](experiments/b200_vllm031/README.md); frozen 0.24 serving
controls and the preserved training runtime retain their original environments.

## Layout

- `src/gleipnir/`: shared code; see the [source map](src/gleipnir/README.md)
  for packages and entrypoints.
- `experiments/<hypothesis>/`: one self-contained hypothesis and its launchers.
- `tests/`: tests grouped by responsibility; see the [test map](tests/README.md).
- `cluster/slurm/`: reusable Slurm entrypoints for Hábrók/RUG.
- `scripts/`: operational and plotting entrypoints, including Lambda Cloud management.
- `figures/`: tracked, reproducible figures and their regeneration commands.
- `outputs/reports/<report_name>/`: self-contained LaTeX reports and proposals
  with committed PDFs; see [the reports index](outputs/reports/README.md).
- `outputs/presentations/<presentation_name>/`: self-contained LaTeX/Beamer
  presentations, with committed public PDFs, local slides, styling, assets and
  build instructions.
- `docs/`: research program, findings, decisions, and infrastructure notes.
- `data/`, `results/`, `logs/`: ignored local artifacts; only `.gitkeep` files are tracked.

Reusable plotting conventions live in `src/gleipnir/analysis/plotting.py`. See
[`figures/README.md`](figures/README.md) for the figure registry and exact
regeneration commands.

Matched monitoring throughput ablations use the config-driven
`gleipnir.campaigns.systems_screen` runner. New systems screens normally need
only a Hydra YAML config and experiment README. Preparation resolves defaults
and overrides into the hashed JSON execution contract; see
[`docs/decisions/config_driven_systems_screens.md`](docs/decisions/config_driven_systems_screens.md).
Full one-epoch monitoring campaigns use `gleipnir.campaigns.monitoring` for
training, FP32 export, BF16 merge, new-adapter parity and registered ID/APPS
evaluation; see the [campaign example](experiments/monitoring_campaign/README.md).
New tool-trajectory training uses strict teacher/source-label agreement
filtering by default; see the [filtering decision](docs/decisions/strict_teacher_filtering.md).
Future merged-weight QAT uses the original arithmetic with lean MLP/GDN work,
MXFP8 forward and BF16 FA4 surrogate backward; see the
[QAT recipe selection](docs/decisions/qat_reference_training.md).
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

Ordinary adapter iteration and training-recipe comparisons use BF16 vLLM 0.31.
Reserve optimized FP8 full-attention projections, FP4 MLP/GDN projections and
MXFP8 attention for QAD-adapted deployment candidates. Use the cached two-logit
monitoring path and
its validated context/pooling corrections; the
[serving decision](docs/decisions/b200_monitor_score_reference.md) defines the
FP4/MXFP8 recipe, supported model/hardware envelope, adapter parity requirements
and checksum-bound controls. Standing launch/lifecycle rules live in the
[inference guide](docs/agent_guides/inference.md).

For activation extraction and steering vectors, use the opt-in
[vLLM Lens monitor integration](experiments/b200_vllm_lens/README.md). It exposes
request-scoped residual capture and steering on the merged monitor in eager
research mode, with a separate numerical baseline.

A cache-free whole-prompt monitoring prototype removes persistent GPU cache
storage. Same-host EU-RO-1 measurements reduce c1 median latency by 15%, with
about 3% lower batch throughput. The earlier NC2 stall remains unresolved.
Keep it experimental and retain the selected serving reference; see the
[cache-free finding](docs/findings/b200_cache_free_serving.md).

Presentation layout and build conventions are documented in
[`outputs/presentations/README.md`](outputs/presentations/README.md).
