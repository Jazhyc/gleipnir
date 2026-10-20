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

| Model | Adapter (FP32 LoRA) | Merged BF16 checkpoint | Mean-OOD pAUROC@20 |
| --- | --- | --- | ---: |
| Gleipnir 4B v0.4 | [`Jazhyc/Gleipnir-4B`](https://huggingface.co/Jazhyc/Gleipnir-4B) | [`Jazhyc/Gleipnir-4B-BF16`](https://huggingface.co/Jazhyc/Gleipnir-4B-BF16) | 0.804 (BF16), 0.770 (optimized) |
| Gleipnir 27B v0.2 | [`Jazhyc/Gleipnir-27B`](https://huggingface.co/Jazhyc/Gleipnir-27B) | [`Jazhyc/Gleipnir-27B-BF16`](https://huggingface.co/Jazhyc/Gleipnir-27B-BF16) | 0.935 (BF16), 0.951 (optimized) |

- **What they are.** MIT-licensed research monitors that score visible
  AI-agent tool-use trajectories for sabotage, exfiltration and other covert
  unauthorized actions. They are not production safety systems.
- **Model cards.** The cards give the scoring prompt, the binary-logit
  interface, results and limitations.
- **Gleipnir 9B** ([`Jazhyc/Gleipnir-9B`](https://huggingface.co/Jazhyc/Gleipnir-9B))
  is superseded by Gleipnir 4B.

## Running inference

Each model scores one fully rendered prompt: the instruction from the model
card, the trajectory, the chat template, then `Prediction:`. The score is the
softmax of the next-token logits for `1` and `0`. There are three ways to
run it:

1. **Transformers + PEFT** (any GPU): load the base model and the adapter
   repository. The model cards have a complete example. Use this to merge or
   fine-tune the adapter.
2. **Merged BF16** (any GPU): load `Jazhyc/Gleipnir-{4B,27B}-BF16` directly
   with Transformers. Its scores are reproducible from run to run.
3. **Optimized B200 serving** (fastest; one NVIDIA B200 only). The image
   quantizes the BF16 checkpoint at load (FP8 attention projections, FP4
   MLP/GDN, MXFP8 attention). One B200 scores about 226K prompt tokens/s
   with the 4B and 54K with the 27B.

   ```bash
   docker run --gpus all -p 8000:8000 \
     -v "$HOME/gleipnir-models:/models" -e GLEIPNIR_MODEL_CACHE=/models \
     ghcr.io/jazhyc/gleipnir-serve:0.1 4b    # or 27b

   curl -s localhost:8000/v1/monitor/score -H 'Content-Type: application/json' \
     -d '{"model": "monitor", "prompt": "<rendered prompt ending in Prediction:>"}'
   # {"score": ..., "margin": ..., "logits": [logit_0, logit_1], "prompt_tokens": ...}
   ```

   - **First start.** The image downloads and verifies the checkpoint (8.8 GB
     or 52 GB) and is ready in about 2.5 minutes (4B) or 4 minutes (27B).
     The `/models` mount keeps the download across restarts.
   - **Prompts.** At most 32,768 tokens.
   - **Mounts.** Do not mount volumes over `/tmp` or `/workspace`, which hold
     the pinned runtime and code; on Runpod, attach volumes elsewhere, for
     example `/models`.
   - **Reproducibility.** Optimized scores vary between runs by about 0.01
     (4B) or 0.006 (27B) per row. Ranking metrics move by about 0.001.
   - **How it was validated:** see
     [`experiments/release_serving`](experiments/release_serving/README.md).

## Evaluation data

- **ID and OOD.** The upstream datasets declare no license, so we don't
  redistribute them. To rebuild the frozen sets locally and check them against
  their pinned SHA256s, run
  `python scripts/build_eval_datasets.py {id,ood,all}`. This needs an
  `HF_TOKEN`, Hugging Face access to the Qwen3-8B and Kimi-K3 tokenizers and
  about 5 minutes once the tokenizers are cached. It makes no paid API calls. `--verify-only` checks existing
  outputs without rebuilding.
- **APPS.** The injection benchmark and our honest-code controls are in the
  public dataset
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
