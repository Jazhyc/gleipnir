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

## Build Pod results (EU-RO-1 B200 `3k1icxdv256bty`, 2026-10-20)

- **Host.** AMD EPYC 9555, driver 595.91.07, volume `cb7bxu71ug`. The stack
  restore took about 2 minutes.
- **Remerge.** Byte-identical to the evaluated checkpoints for both models
  (source shards, every output file, projection count).
  - The 4B merged the verified local copy of the QAD adapter (SHA-256
    `4eec7b62…`), because the volume copy was lost.
  - Uploaded privately, untagged: `Jazhyc/Gleipnir-4B-BF16` (commit
    `3599d85`) and `Jazhyc/Gleipnir-27B-BF16` (commit `8717723`).
- **Startup.** `serve.py` reached ready on the first attempt, with warm
  caches: 4B in 60–126 s, 27B in 60 s.

### Validation passes

Sequential batches of 128 rows, as in the recorded campaign. The first 4B
attempt sent the whole workload through one connection pool; it dropped a
connection on OOD and does not mirror the campaign batching.

| Model, split | pAUROC@20 pass a / b (recorded) | Pass a vs recorded: MAE, corr | Pass a vs pass b, same host: MAE, corr | Throughput |
|---|---|---|---|---:|
| 4B ID | 0.9119 (0.9117) | 0.0112, 0.9970 | 0.0110, 0.9970 | 219K tok/s |
| 4B OOD | 0.7700 (0.7703) | 0.0092, 0.9959 | 0.0090, 0.9965 | 226K tok/s |
| 27B ID | 0.9447 / 0.9434 (0.9444) | 0.0072, 0.9990 | 0.0065, 0.9991 | 53.6K tok/s |
| 27B OOD | 0.9495 / 0.9508 (0.9508) | 0.0061, 0.9990 | 0.0057, 0.9992 | 53.3K tok/s |

For the 4B, the pass a / b column comes from the two repeat passes; the
pre-repeat pass gave 0.9119 ID and 0.7700 OOD.

- **The frozen per-row gates fail on every split.** The gates were MAE ≤ 0.005
  and correlation ≥ 0.999.
- **The cause is the stack's own nondeterminism, not the image or the
  host.** Two back-to-back passes on the same server differ from each other as
  much as either differs from the recorded run (ratio 1.02–1.11).
  - The gates were copied from the parity-canary limits without first
    measuring the stack's repeatability.
  - The archived vLLM 0.31 repeats show the same scale: pairwise MAE
    0.002–0.008 and correlation 0.9985–0.9995 on 320 rows.
- **The ranking gate passes on every split** (|Δ pAUROC@20| ≤ 0.0013).
- **Image contents** (build time): 51 receipts traced from `results/`, none
  from `data/`. The receipts contain only commands, hashes, IDs and scores.

## Fresh-Pod validation (US-NE-1 B200 `bpelqt8zaoq51q`, 2026-10-20)

- **Host.** A fresh Pod from the image, with no volume: Intel Xeon 6960P. The
  image was digest `2b537310…` before the cache fix below.
- **First start failed** in vLLM's JIT warm-up with `KeyError: 'cubin'`. The
  `open()` trace sees compile-cache index files but not the cubins that
  Inductor and Triton load natively, so the image carried incomplete cache
  entries.
- **Fix: ship no compile caches.**
  - `image/clear_compile_caches.py` appends an opaque-whiteout layer that
    empties the vLLM, Triton and Inductor compile caches.
  - The small native caches (FlashInfer, cuDNN frontend, CuTe DSL) shipped
    whole and stay.
  - Final image: `ghcr.io/jazhyc/gleipnir-serve:0.1`, digest `15673c65…`.
- **Cold-start timings** (entrypoint to ready, first attempt):
  - 4B: 134 s, with the download already cached.
  - 27B: 327 s, of which about 175 s downloads and verifies 52 GB; init took
    39 s, including 22 s of compilation.
- **4B passes on this host:**

  | Split | pAUROC@20 a / b (recorded) | Pass a vs pass b: MAE, corr | Pass a vs recorded: MAE, corr | Throughput |
  |---|---|---|---|---:|
  | ID | 0.9110 / 0.9121 (0.9117) | 0.0102, 0.9976 | 0.0122, 0.9968 | 202–215K tok/s |
  | OOD | 0.7693 / 0.7698 (0.7703) | 0.0088, 0.9964 | 0.0104, 0.9955 | 210K tok/s |

## Gate revision (user decision, 2026-10-20)

The frozen per-row gates (MAE ≤ 0.005, correlation ≥ 0.999 against the
recorded rows) fail on every split, including same-server reruns. They are
kept above as the pre-registered result. The user accepted replacement gates
measured against the stack's own same-host repeat noise. Each split must
meet all of these:

1. image-vs-recorded MAE ≤ 1.25 × the host's pass-a-vs-pass-b MAE;
2. image-vs-recorded correlation ≥ the host's repeat correlation − 0.001;
3. |Δ source-macro pAUROC@20| ≤ 0.005, unchanged.

The model cards disclose that optimized-serving scores vary between runs by
about 0.01 per row for the 4B and about 0.006 for the 27B, while ranking
metrics move by about 0.001.

## Final image check and release (2026-10-20)

- **Final image check.** A fresh Pod (US-TX-6, Intel Xeon 6972P) ran the
  final digest `15673c65…`. The compile caches came up empty, as the
  whiteout layer intends.
  - **4B:** ready in 139 s, including its 8.8 GB download. pAUROC@20 ID
    0.9119 and OOD 0.7700 (recorded 0.9117 and 0.7703); MAE 0.0119 and
    0.0097; 205–213K tok/s.
  - **27B:** ready in 219 s, including its 52 GB download. ID 0.9450
    (recorded 0.9444); MAE 0.0072; 53.9K tok/s.
- **Revised gates.** All 15 split-passes with a same-host repeat pass:
  - **MAE ratio:** image-vs-recorded MAE is 1.00–1.20× the host's own
    repeat MAE.
  - **Correlation:** within 0.001 of the repeat correlation.
  - **Ranking:** |Δ pAUROC@20| ≤ 0.0019.
  - **Splits without a repeat.** The fresh-host 27B OOD (its pass b was cut
    short when the Pod was terminated) passes against the build-host
    repeat: MAE 0.0060 against 0.0057. The final-image passes are single
    passes; they match the other hosts' agreement levels.
- **Published:**
  - `Jazhyc/Gleipnir-4B@v0.4` and `Jazhyc/Gleipnir-27B@v0.2` (updated cards,
    unchanged adapters);
  - `Jazhyc/Gleipnir-4B-BF16@v0.4` and `Jazhyc/Gleipnir-27B-BF16@v0.2`
    (public; weight hashes match `release_manifest.json` anonymously).
- **Image visibility.** The GHCR package `gleipnir-serve` is private until
  its visibility is changed in the GitHub package settings; the GitHub API
  cannot change it.
