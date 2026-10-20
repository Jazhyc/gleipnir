# Release packaging for the optimized B200 serving stack

Packages Gleipnir 4B **v0.4** and Gleipnir 27B **v0.2** for outside users.
These are packaging releases: the weights are unchanged.

- **Hugging Face, two repos per model.**
  - `Jazhyc/Gleipnir-4B` and `Jazhyc/Gleipnir-27B` keep the FP32 and `vllm/`
    adapters, for merging and fine-tuning.
  - `Jazhyc/Gleipnir-4B-BF16` and `Jazhyc/Gleipnir-27B-BF16` hold the merged
    BF16 checkpoints, for stock vLLM or Transformers and for the image.
  - Every repo of a model gets the model's version tag.
- **No pre-quantized repo.** The optimized stack quantizes the BF16
  checkpoint online at load: NVFP4 MLP/GDN, FP8 attention projections and
  MXFP8 attention.
- **Serving image** `ghcr.io/jazhyc/gleipnir-serve` (image version 0.1). It
  is a frozen snapshot of the validated stack, not a redesign. It contains:
  - both serving runtimes and the native FA4, MXFP8 and Triton trees;
  - the receipts the integrity checks bind;
  - warm compiler caches;
  - the repository source, at the paths the receipts record.

  `image/build.py` assembles it with `crane`, without a Docker daemon. The
  receipt allowlist comes from an `open()` audit trace of real launches
  (`image/open_trace.py`), so no dataset-derived rows enter the image.

## Components

- `targets.py`: one validated configuration per model. Each replays the
  recorded server command of the run whose optimized scores the card reports:
  4B `qad-v02-optimized01`, 27B `opt27b_teacher_eu01`.
- `remerge.py`: rebuilds the merged checkpoint at the recorded path and
  requires byte identity with that run's `merged_artifact.json`.
- `fetch.py`: downloads `*-BF16@<version>` to the recorded merged path. The
  path is part of the compiler-cache key. Every file is verified against the
  recorded hashes.
- `serve.py`: replays the recorded command with the recorded environment.
  - **27B sources:** for the 27B it first restores the repository versions
    of the four diagnostic sources that the 4B launch rewrites.
  - **Compile race:** it retries the intermittent `TYPE_UNSTABLE_JOIN`
    compile race, up to three attempts.
- `validate.py`: scores a target's frozen ID and OOD workloads against a
  running server.

## Validation (frozen before the build, 2026-10-20)

A fresh Pod is launched from the pushed image, with no network volume. For
each model it fetches the checkpoint, serves it, and runs `validate.py` on the
full ID (3,012) and OOD (6,395) workloads. Each split must meet all of these
against the recorded optimized rows:

- per-row score MAE ≤ 0.005;
- correlation ≥ 0.999;
- |Δ source-macro pAUROC@20| ≤ 0.005.

The recorded values are 4B ID 0.9117 and OOD 0.7703; 27B ID 0.9444 and OOD
0.9508. Prompt throughput is reported against the recorded 226K (4B) and
53.9K (27B) tokens/s, but is not gated.

- **Release:** HF tags and making the image public follow a passing
  validation.
- **Stop conditions:** a failed gate, a startup failure other than the
  known compile race, a remerge identity failure, or traced dataset rows.
