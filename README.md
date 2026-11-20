# Gleipnir

Gleipnir is a research project on monitors for AI control: small models that
read the visible trajectory of an AI agent (its messages, tool calls and tool
outputs) and score how likely it is that the agent did something problematic,
such as sabotage, data exfiltration or bypassing a safeguard.

A Gleipnir monitor writes no rationale. It needs one forward pass over the
trajectory and reads its score from the logits of the tokens `0` and `1`, so
it can run before every action of an agent. The monitors are LoRA adapters on
Qwen3.5-4B and Qwen3.8-27B, trained on the soft judgments of a larger teacher
model.

The [technical report](outputs/reports/gleipnir_v1_technical_report/) (in
preparation) describes the method, the evaluation and the serving stack. The
repository is experiment-centric: each hypothesis has its own directory under
`experiments/`, reusable code lives in `src/gleipnir/`, and durable
conclusions live in `docs/`.

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
2. **Merged BF16 with stock vLLM** (any GPU; the throughput route off the
   B200). Serve `Jazhyc/Gleipnir-{4B,27B}-BF16` with vLLM 0.31's pooling
   runner, which turns the model into a classifier over the `0` and `1`
   logits at the last position. Use the 27B repository and `v0.2` for the
   27B. Transformers also loads these repositories directly.

   ```bash
   vllm serve Jazhyc/Gleipnir-4B-BF16 --revision v0.4 --served-model-name monitor \
     --dtype bfloat16 --language-model-only --max-model-len 32768 --no-enable-prefix-caching \
     --runner pooling --convert classify \
     --pooler-config '{"pooling_type": "LAST", "use_activation": false}' \
     --hf-overrides '{"classifier_from_token": ["0", "1"], "method": "no_post_processing",
       "num_labels": 2, "is_causal": true, "text_config": {"classifier_from_token": ["0", "1"],
       "method": "no_post_processing", "num_labels": 2, "is_causal": true, "mamba_ssm_dtype": "float32"}}'

   curl -s localhost:8000/classify -H 'Content-Type: application/json' -d '{"model": "monitor",
     "input": "<rendered prompt>", "add_special_tokens": false, "use_activation": false}'
   # data[0].probs holds the raw [logit_0, logit_1]; score = softmax(...)[1]
   ```

   - **Overrides.** The classifier settings repeat inside `text_config`,
     because Qwen3.5/3.8 wrap the language model; `mamba_ssm_dtype` keeps
     the GDN state FP32.
   - **Reproducibility.** This recipe reproduces the recorded BF16 scores
     to about 0.005 per row
     ([check](experiments/release_serving/README.md)).
   - **Prefix caching** stays off, matching the evaluated setup.
3. **Optimized B200 serving** (fastest; one NVIDIA B200, or a B300). The image
   quantizes the BF16 checkpoint at load (FP8 attention projections, FP4
   MLP/GDN, MXFP8 attention). With one request at a time, one B200 scores an
   agent trajectory of about 10,000 tokens in 48 ms with the 4B (205K prompt
   tokens/s) and in 192 ms with the 27B (51K). Sending 8 requests at once
   raises the throughput to 230K and 53K and the latency about ninefold
   ([measurements](experiments/serving_concurrency/README.md)).

   ```bash
   docker run --gpus all -p 8000:8000 \
     -v "$HOME/gleipnir-models:/models" -e GLEIPNIR_MODEL_CACHE=/models \
     ghcr.io/jazhyc/gleipnir-serve:1.2 4b    # or 27b

   curl -s localhost:8000/v1/monitor/score -H 'Content-Type: application/json' \
     -d '{"model": "monitor", "prompt": "<rendered prompt ending in Prediction:>"}'
   # {"score": ..., "margin": ..., "logits": [logit_0, logit_1], "prompt_tokens": ...}
   ```

   - **First start.** The image downloads and verifies the checkpoint (8.8 GB
     or 52 GB) and is ready in about 2 to 5 minutes (4B) or 5 to 8 minutes
     (27B), most of it download. The `/models` mount keeps the download
     across restarts, after which a start takes about 2 minutes.
   - **Since image 1.0** the 4B captures CUDA graphs for prompts of up to
     4,096 tokens and both models tokenize with Gigatoken. Neither changes a
     score at one request at a time; `--recorded-graphs` and
     `--transformers-tokenizer` switch them off.
   - **Since image 1.1** the image also runs on an NVIDIA B300, for
     flexibility: it is about as fast as on the B200 and costs more per
     hour. Scores match the B200's with rare exceptions (1 of 248 check
     prompts for the 27B, none for the 4B). On a B200, 1.1 behaves as 1.0.
   - **Since image 1.2** the image is easier to reach on hosted GPUs, and
     serving is unchanged. Any command other than `4b`/`27b` runs as given
     (e.g. `bash`); on Runpod, set `PUBLIC_KEY` and the start command `4b`
     to get SSH beside the server, or `/start.sh` for a shell only. `/tmp`
     is writable by all users, so `apt-get` works, and the layers are zstd
     compressed (15.3 GB), which unpack faster on a first pull.
   - **Prompts.** At most 262,144 tokens, the checkpoints' native range
     (`--max-model-len 32768` restores the 0.1 limit). Image 0.1 serves at
     most 32,768.
   - **Prefix caching** is off. Append `--prefix-caching` (e.g. `... 4b
     --prefix-caching`) to rescore a growing session faster; cached scores
     can differ from uncached ones by up to about 0.15, so keep it off for
     evaluation.
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

## Development

```bash
cp .env.example .env    # names of the credentials; fill in only what you need
./setup_dev.sh
source .venv/bin/activate
pytest
ruff check .
```

The lock file pins the environment (Python 3.12, vLLM 0.31.0, Transformers
5.14.1). The full environment needs a CUDA GPU. Experiments run on rented
B200 GPUs; [`docs/infrastructure.md`](docs/infrastructure.md) records how.

## Layout

- `src/gleipnir/`: shared code; see the [source map](src/gleipnir/README.md)
  for packages and entrypoints.
- `experiments/<hypothesis>/`: one self-contained hypothesis with its README,
  configuration, entrypoint and tests.
- `tests/`: tests grouped by responsibility; see the [test map](tests/README.md).
- `scripts/`: operational entrypoints (Runpod and Lambda Cloud management,
  dataset builds, releases, plotting).
- `cluster/slurm/`: Slurm launchers from the university cluster on which the
  project started.
- `figures/`: tracked, reproducible figures; see the
  [figure registry](figures/README.md) for the regeneration commands.
- `outputs/reports/`: self-contained LaTeX reports with committed PDFs; see
  [the reports index](outputs/reports/README.md).
- `outputs/presentations/`: self-contained presentations; see
  [their README](outputs/presentations/README.md).
- `model_releases/`, `data_releases/`: tracked inputs of the Hugging Face
  releases (model and dataset cards, release configuration).
- `docs/`: research program, findings, decisions and infrastructure notes.
- `data/`, `results/`, `logs/`: ignored local artifacts; only `.gitkeep`
  files are tracked.

## Where to read next

- [Research program](docs/research_program.md): goals, scientific direction
  and the evaluation contract. Start here.
- [Documentation index](docs/README.md): findings and decisions, newest first.
  `docs/findings/` and `docs/decisions/` hold all of them, including older
  infrastructure records that the index does not list.
- [Repository guidelines](AGENTS.md) and the task guides for
  [training](docs/agent_guides/training.md),
  [inference and evaluation](docs/agent_guides/inference.md) and
  [compute](docs/agent_guides/compute.md): the working rules and the
  currently selected recipes.
- The README of the experiment you are changing.
- [vLLM Lens integration](experiments/b200_vllm_lens/README.md): activation
  capture and steering on the merged monitor, for interpretability work.
