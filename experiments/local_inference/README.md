# Local merged Gleipnir 4B baseline

## First optimization: 4,096-token prefill budget

Hypothesis: doubling `max_num_batched_tokens` from 2,048 to 4,096 improves
prefill efficiency by reducing chunk/scheduling overhead. Change only this
engine setting; retain the frozen 32-row order, weights, prompt, two concurrent
sequences, cache settings, canaries, and one timed pass. Compare against
`baseline32` without rerunning it. This is a development screen, not a held-out
quality claim or automatic promotion. Report paired score and margin drift,
threshold flips, metrics, and thermal conditions; stop on existing parity gates,
OOM, missing/nonfinite outputs, truncation, or artifact drift. No new numerical
equivalence tolerance or repeat-noise estimate is claimed.

Run `.venv/bin/python -m experiments.local_inference.run --config
experiments/local_inference/prefill4096.json`. Results are saved separately to
`results/local_inference/prefill4096/`, with campaign status in
`results/local_inference/prefill4096_status.json`. Existing results are protected.

Measured result (2026-09-23): **no demonstrated meaningful speedup**. One pass
scored all 338,780 tokens in 35.4894 seconds versus baseline 35.9089 seconds:
9,545.95 versus 9,434.42 prompt tokens/s, a 1.0118x throughput ratio (+1.18%).
With one pass and differing thermal histories, this small difference is
inconclusive; retain 2,048 as the reference, not an automatic promotion.

Paired score drift: mean absolute 0.005636, maximum 0.030967, p95 0.029793,
correlation 0.999574, zero threshold flips. Maximum absolute drift among the six
baseline scores in [0.4, 0.6) was 0.030967. Macro AUROC changed
0.921488 -> 0.925620; pAUROC@20 0.756198 -> 0.776860; Brier
0.097959 -> 0.099486. These small-set ranking changes are not evidence of a
better judge; numerical equivalence is not established. Unique scores were
30 -> 31. Both serving canary comparisons passed, with unchanged input/model
identities, and the longest input completed without OOM or truncation.

Whole-process time was 172.51 seconds versus 85.30 seconds. Construction took
107.48 seconds versus 24.07 seconds: this candidate needed a new compiled graph
(22.63 seconds torch.compile) and an initial profiling/warmup run (59.85 seconds),
whereas the baseline reused existing compilation caches. Do not attribute this
startup difference to steady-state scoring. Canary warmup was 6.59 seconds.
No second run was made to measure cached candidate startup.

All three scoring telemetry samples showed software thermal throttling at 84 C
and SM clocks of 2,595–2,670 MHz, compared with baseline 85–87 C. Sampled GPU
memory was 14,451 MiB (whole-device usage); vLLM reported 3.39 GiB cache capacity
or 104,261 tokens. The worker and telemetry process exited successfully.

Reproduce the paired audit with `.venv/bin/python -m
experiments.local_inference.compare_runs results/local_inference/baseline32
results/local_inference/prefill4096`. `comparison.json` retains per-row scores,
score/margin deltas, flips, per-source drift, baseline-score bins, and metrics.
Raw logprobs and launch/engine settings remain in each condition's artifacts.

## 32-row iteration workload

The user selected 32 rows for fast iteration after the initial 512-row baseline.
`iteration32.json` freezes four STRIDE negatives, six STRIDE positives, and
eleven examples of each Gloom label, selected from the existing 512 rows by the
same seeded source/label/length-bin procedure. The slice contains 338,780 prompt
tokens (2,379–27,544 per row). Preserve the 512-row artifact for occasional
broader checks. Do not interpret small-set AUROC as a precise quality estimate;
retain paired logprobs, margins, scores, threshold flips, and source coverage.

Hypothesis: the smaller workload gives a substantially faster complete iteration,
although engine startup may dominate. The intervention changes only dataset
size: retain the merged weights, serving settings, four original parity rows,
one longest-selected-row warmup, and one timed pass. Existing parity and failure
gates still apply. Measure engine initialization, canary warmup, scoring, and
whole-process wall time separately. Compilation caches are reused naturally;
this is a fresh-process measurement, not a deliberately cold-cache experiment.

Run `.venv/bin/python -m experiments.local_inference.run --iteration32`.
Inputs are `data/local_inference/iteration32.jsonl` and its manifest. Results
are `results/local_inference/baseline32/`; campaign status is
`results/local_inference/iteration32_status.json`. The ordinary invocation
retains the historical 512-row preparation workflow. To record thermals during
this run, invoke `telemetry --output results/local_inference/baseline32 --status
results/local_inference/iteration32_status.json` through the experiment module.

Measured on 2026-09-23, one pass completed with the following wall times:

| Phase | Seconds |
| --- | ---: |
| Main-function preparation/imports and artifact validation | 15.66 |
| vLLM construction (`LLM(...)`, including worker startup) | 24.07 |
| Parity and longest-input canary warmup | 4.48 |
| 32-row scoring, including rendering and output extraction | 35.91 |
| Complete benchmark subprocess, including shutdown | **85.30** |

The total includes initial module imports and shutdown outside the separately
timed main-function phases. Throughput is **9,434.42 prompt tokens/s**. Cached
compiled graphs loaded in 0.859 seconds; vLLM's narrower internal engine-init
timer was 6.98 seconds, compared with 82.48 seconds in the earlier startup.
Thus total fixed overhead is material, but cached initialization is much faster
than the earlier estimate based on the initial setup. Changing compilation
settings can produce cache misses and increase startup again.

All 32 saved identities, labels, prompt hashes, token counts, normalized scores,
and metrics were independently checked. Relative to the same rows' existing
512-row scores, mean absolute score change was 0.005573, maximum 0.030967,
correlation 0.999578, and there were zero 0.5-threshold flips. This comparison
changes scheduling/order and run conditions, not model weights. Use the fresh
32-row predictions as the iteration reference. Small-set macro pAUROC@20 was
0.756198 and AUROC 0.921488; these are not comparable with full-512 metrics as
evidence of degradation because the population differs.

Thermal samples during scoring showed 85–87 C and active software thermal
throttling; the GPU worker exited successfully. No repeat-noise estimate is
claimed. Input SHA-256:
`aadb48556b134150e43946ba39d31512498e2d61b8a88a7a24fd9617b5633f03`.
The measured totals are in `baseline32/result.json` and
`baseline32/process_timing.json`, alongside the single-pass predictions.

Hypothesis: a BF16 merge of the released Gleipnir 4B LoRA provides an efficient
fixed-weight baseline for later vLLM serving optimizations on the local RTX 4080.
The initial baseline used a frozen 512-row ID subset, retained for occasional
broader checks after selecting 32 rows for iteration. Neither full-ID
confirmation nor strict-OOD selection is in scope.

Reconstruct CoT-removed trajectories from the checksum-pinned STRIDE and Gloom
sources using the existing ID loaders and source-aware CoT remover. Preserve the
compact student prompt and non-thinking `Prediction:` boundary. Stratify by
source, hard label, and length quartile, then select by seeded ID hash. Freeze
membership before scoring. Source labels are evaluation-only; no teacher calls.
All source datasets remain ignored and must not be redistributed (the existing
source inventory records unspecified upstream licenses).

The frozen subset has 5,760,843 prompt tokens, ranging from 1,291 to 29,475
tokens per row, with SHA-256
`f5800ce52b38184bf3854fbf8e7a91257a3774c2f0a859c598c97e26f5829ebf`.
Reconstruction matches the recorded 3,012-row population's 33,750,959 prompt
tokens; its JSONL metadata schema is new, so it is not claimed to match the
historical full-file hash. The four short eager canaries have lengths
1,291, 1,354, 1,716, and 2,195; no prompt is shortened for a canary.

Download the published master adapter at a resolved immutable Hub revision and
the pinned Qwen3.5-4B base. Keep the FP32 master unchanged. Merge on CPU with
PEFT safe merging, save BF16 inference weights, and hash every artifact. A
bounded Transformers canary compares unadapted base, FP32-master adapter, and
merged weights; a matched vLLM canary then checks serving agreement. These
eager canary includes one short actual subset row per source/label pair; vLLM
also runs the longest subset input as a separate memory/kernel canary. Require
nonzero adapter effect, mean absolute score error <= 0.02, maximum error <= 0.10,
and correlation >= 0.99 when scores have nonzero variance. Report all flips;
these are baseline-construction tolerances, not future optimization tolerances.
Stop before full-subset evaluation if a gate fails.

Use one persistent vLLM engine, one constrained decision token, explicitly
requested 0/1 logprobs, and one warmed timed pass over the fixed input order.
Disable prefix caching initially to avoid replay-dependent timings. Record raw
decision logprobs, margins, scores, per-source/macro ranking and calibration,
threshold diagnostics, ties, initialization time, and prompt tokens/s.
Stop on missing or nonfinite scores, OOM, truncation, artifact/hash drift, or a
failed parity gate. Do not change data or quality criteria in response to scores.
Future optimizations will be compared against this merged baseline on these
same rows. Per the user's update during the first pass, use one measurement per
condition and target meaningful speedups; repeat noise is not estimated. Small
speed differences should not be interpreted as improvements. Keep paired score
drift diagnostics even with single-pass timing.

The workstation has a 16 GB RTX 4080 and 47 GiB RAM. No Slurm or cloud capacity
is involved. No in-chat scheduling tool is available in this session; monitoring
uses active-turn startup checks and continued active-turn checks during scoring,
and cannot promise agent follow-ups after the turn ends.

Startup exposed two launcher/build issues: explicit Python invocation did not
put `.venv/bin/ninja` on PATH, and FlashInfer's sampling Ninja file does not
escape the space in this checkout's `AI Control` directory. The runner now
adds its executable directory to PATH. The frozen environment sets
`VLLM_USE_FLASHINFER_SAMPLER=0`; this workload uses greedy one-token decisions,
and vLLM's native sampling path avoids that optional JIT dependency. FlashAttention
2 and Triton/FLA GDN remain the automatic model-kernel choices on SM89. Both
failed startup attempts and logs are retained separately.

Provenance: reuse `prepare_qwen_reasoning_id_benchmark` source loaders,
`id_cot_only_evaluation.prepare.strip_cot`, the shared student prompt, and shared
binary metric functions. This is a separate local campaign; historical configs
and result contracts are unchanged.

## Run

```bash
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/python -m experiments.local_inference.fetch
.venv/bin/python -m experiments.local_inference.run
```

The runner prepares the subset, merges the release, checks the eager reference,
and starts the vLLM baseline sequentially. Each phase has a separate log under
`logs/local/local_inference/`; status and results are in
`results/local_inference/`. Failed attempts must be retained before rerunning
the baseline; completed timed passes are not silently overwritten or resumed.

For local temperature/clock diagnostics, run
`.venv/bin/python -m experiments.local_inference.telemetry` in a second process
after `baseline/status.json` exists. It samples every ten seconds and exits when
the campaign completes or fails. This is instrument logging, not scheduled agent
monitoring. The first baseline showed software thermal throttling at 87 C;
compare future configurations under comparable cooling and record telemetry.
Sampling began partway through its first pass, so those samples are not a full
startup or first-pass thermal trace.

## Baseline result (2026-09-23)

One complete pass processed all 512 rows and 5,760,843 prompt tokens in
**631.65 seconds (10m32s)**: **9,120.28 prompt tokens/s**, about 0.81 rows/s.
This is wall time around `LLM.generate`, including its input rendering and
output extraction; it excludes initialization and canary warmup. Prefix caching
was disabled. The run used two concurrent sequences and a 2,048-token prefill
budget. No repeated-pass noise estimate was made.

| Source | Rows | pAUROC@20 | AUROC | Brier |
| --- | ---: | ---: | ---: | ---: |
| STRIDE | 161 | 0.967930 | 0.993197 | 0.038498 |
| Gloom-Exfiltration | 351 | 0.752760 | 0.923961 | 0.121559 |
| Source macro | 512 | 0.860345 | 0.958579 | 0.080028 |

The 512 predictions contain 397 distinct scores (115 extra tied rows). These
are subset results, not a measured improvement over the historical full-ID
metrics. All IDs, sources, labels, prompt hashes, token counts, decision-logprob
normalizations, throughput, and aggregate metrics were independently rechecked
from the saved first-pass artifacts.

The four-row eager merge gate had mean absolute score error 0.004477 and maximum
0.017909 against the original FP32 master. vLLM versus master had mean error
0.008683, maximum 0.030490, and correlation 0.999590; neither canary comparison
flipped a decision at 0.5. The longest 29,475-token subset prompt also passed.
This is bounded canary evidence, not a full-subset master-versus-merge comparison.

The GPU was thermally limited: all 47 recorded first-pass telemetry samples
reported software thermal throttling, with peak temperature 89 C and SM clocks
2,520–2,655 MHz. Hardware thermal slowdown was not observed in these samples.
Use comparable cooling/thermal conditions for future timing comparisons.

The initial runner had three passes configured; the user reduced the scope
during pass one. A task-specific watcher stopped the worker immediately after
`repeat_0.json` was saved. Only one completed pass is retained. The original
launch config and interruption log are preserved; `summarize.py` audits and
finalizes this specific saved-pass recovery. The interruption therefore does
not represent an inference failure. The checked-in config now requests one
pass directly, and single-pass repeat stability is recorded as null.

Artifacts: `results/local_inference/baseline/result.json`,
`predictions_0.json`, `repeat_0.json`, `serving_parity.json`,
`longest_canary.json`, `engine_config.json`, and `gpu_telemetry.jsonl`.
The merged model is `results/local_inference/merged_bf16/`; the original master
remains in `results/local_inference/adapter/`. Published adapter revision:
`411bdb7cf28bf153d03820f242c7f0668762fa6d`.
