# Local Gleipnir 4B inference campaign preparation

Prepared 2026-09-23. The user selected a subset-only campaign and a merged-BF16
baseline; executable work now lives in
[`experiments/local_inference`](../../experiments/local_inference/README.md).

## Machine and environment

The workstation exposes one NVIDIA GeForce RTX 4080 (16,376 MiB), an AMD
Ryzen 9 5900X (24 logical CPUs), and 47 GiB host RAM through WSL. At inspection,
1,641 MiB GPU memory was already occupied. Keep display headroom and record
other GPU activity with each measurement.

Installed and verified: Python 3.12.13, PyTorch 2.11.0+cu130, Transformers
5.14.1, and vLLM 0.24.0. The vLLM engine imports and a finite BF16 CUDA
matrix multiplication passed on compute capability 8.9. Ten focused prompt/CoT
tests and repository Ruff checks passed, as did an offline bootstrap rerun.
This does not yet verify Qwen loading, its serving kernels, or adapter parity.

`setup_dev.sh` now supports workstations without environment modules or a
system `python` command. It selects Python 3.12 through uv and keeps local
caches inside ignored `.cache/`; module-based cluster defaults are preserved.

```bash
./setup_dev.sh
source .venv/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
```

Use the checked-in lock and the released FP32 master plus its vLLM serving
adapter. The pinned base is `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Preserve the frozen compact prompt,
non-thinking boundary, and normalized next-token `0`/`1` score.

## Dataset choice

Use the [canonical CoT-removed ID set](../decisions/cot_removed_id_validation.md),
which preserves visible assistant prose. It contains 946 STRIDE and 2,066
Gloom-Exfiltration rows. The recorded complete workload is **33,750,959 prompt
tokens**, or 11,205.5 per example, plus 3,012 one-token decisions. This is a
historical exact count, not a new local tokenization.

The canonical prompts and manifest are absent from this workstation checkout.
Restore them with the decision's checksums, or reconstruct them from the pinned
upstream inputs and verify identical prompt bytes before freezing a subset.
The existing preparation chain starts with
`prepare_qwen_reasoning_id_benchmark.py`, then `prepare_teacher_id_cache.py`,
`prepare_distillation_id.py`, and the CoT removal in
`experiments/id_cot_only_evaluation/prepare.py`. The last historical entrypoint
also expects archived job metadata; do not run it blindly or rewrite historical
configs merely to obtain the dataset. Model weights are also absent locally.

For repeated screening, freeze **512 rows**, proportional to source and label:
63 STRIDE negatives, 98 STRIDE positives, 176 Gloom negatives, and 175 Gloom
positives. Within each group, allocate across prompt-length quartiles and select
deterministically by a seeded hash of ID; record IDs, prompt hashes, tokenizer
revision, and exact token counts. Do not truncate long examples to fit memory.
The proportional estimate is **5,737,215 prompt tokens**; the actual selection
will differ. Keep a separate longest-input startup canary. Per the user's
updated scope, use only these 512 rows throughout, including confirmation;
leave full-ID and strict OOD outside this campaign.

## Measurement and drift

Hypothesis: serving changes improve steady-state prefill throughput while
preserving the same judge's continuous scores and decisions. Establish a BF16
merged baseline from the published LoRA, first checking original-master/merge
and serving parity. Future serving interventions use this merged reference.
Start conservatively on 16 GB: memory utilization
0.80, maximum sequences 2, batched-token budget 2,048, chunked prefill, and
32,768 context. These are proposed canary settings, not a verified fit or an
optimal configuration. Verify supported kernels and master/serving parity
including a nonzero adapter effect before scaling.

Per the user's updated scope, take one baseline pass and one pass per candidate;
target meaningful speedups and do not claim a repeat-noise estimate. Warm up
before timing, disable prefix caching, retain identical ordering,
and report initialization separately. Measure wall time, rows/s, prompt tokens/s,
peak memory, and all paired logit margins and scores. Compare absolute score
deltas, threshold flips at 0.5, ties, per-source and source-macro pAUROC@20,
AUROC, and Brier. Also inspect baseline-margin bins so saturated easy examples
do not hide drift near the decision boundary. Use paired, lineage-grouped
resampling for quality comparisons where grouping metadata is available.

Historical BF16 runs had nonzero repeat variation, so do not interpret every
changed score as optimization damage or select solely on aggregate AUROC.
Freeze numerical and quality tolerances before testing candidates, acknowledging
that the single-pass baseline does not estimate repeat noise. A 512-row screen
can reveal paired drift but
cannot certify absence of small population-level quality loss. Stop on OOM,
truncation, missing/nonfinite scores, contract drift, or a failed serving-parity
canary. Confirm repeatability on the frozen subset before choosing a candidate.

## Runtime budget

For this one-token judge, prompt processing dominates. Time is prompt tokens
divided by measured prompt tokens/s; generated-token throughput is misleading.
The following are planning scenarios, **not measured RTX 4080 performance**:

| Sustained prompt tokens/s | 512-row estimate | Complete 3,012 rows |
| ---: | ---: | ---: |
| 5,000 | 19.1 min | 112.5 min |
| 10,000 | 9.6 min | 56.3 min |
| 20,000 | 4.8 min | 28.1 min |

Add model loading, compilation, warmup, and serialization. The updated campaign
uses one timed pass per condition; reference and candidate both require passes.
For context, the historical H100 dynamic-LoRA pass processed 34,631,573 tokens
in 861.24 seconds (40.2k tokens/s), but that is not a workstation estimate.
Replace these scenarios with a stratified local pilot once inputs and weights
are staged, then calculate the estimate from its measured length mix.

## Measured subset baseline

The merged-BF16 baseline is now complete on the fixed 512 rows: **5,760,843
prompt tokens in 631.65 seconds**, or **9,120.28 prompt tokens/s**. The user
selected one pass throughout; no repeat-noise estimate is claimed. Macro
pAUROC@20 is 0.860345 and macro AUROC is 0.958579 on this subset. Software
thermal throttling was active in all 47 recorded first-pass samples, so this
timing describes the workstation under its current cooling conditions.
See the [experiment result](../../experiments/local_inference/README.md) for
parity checks, per-source metrics, retained artifacts, and startup fixes.

The user subsequently selected a 32-row stratified slice for iteration. Its
338,780 tokens took **35.91 seconds** in the measured single pass, or **9,434.42
tokens/s**. A fresh benchmark subprocess with existing compilation caches took
**85.30 seconds total**, including **24.07 seconds constructing vLLM** and
**4.48 seconds of canary warmup**; the remainder includes validation/imports,
scoring, persistence, and shutdown. Use `iteration32.json` and `baseline32/`
for iteration, while preserving the 512-row artifacts for broader checks.

## First prefill-budget screen

Increasing only `max_num_batched_tokens` from 2,048 to 4,096 on the frozen 32
rows yielded **35.49 vs 35.91 seconds**, or **9,545.95 vs 9,434.42 tokens/s**.
The 1.18% throughput difference is not a meaningful demonstrated win from one
pass under thermal throttling. Keep the original baseline as the reference.
Mean/max absolute score drift was 0.005636/0.030967 with zero threshold flips;
this is not proof of numerical equivalence or population-level quality parity.
The candidate's 172.51-second total included new compilation and initial engine
profiling, unlike the cached 85.30-second baseline. No extra pass was run.
Full paired evidence and thermal/startup caveats are in the experiment README
and `results/local_inference/prefill4096/comparison.json`.

## GPU profiling finding

A bounded 16-step baseline prefill trace now works by initializing CUPTI inside
the worker before model loading/graph capture. Late initialization repeatedly
failed; no driver or permission changes were needed for activity tracing.
The trace contains 12,944 kernels: GEMM/GEMV consumes 75.51% of summed GPU
kernel time, FlashAttention 11.21%, named gated-delta/convolution kernels 7.29%,
KV writes/bookkeeping 0.12%, and other kernels 5.86%. Kernel activity occupies
98.85% of the sampled first-to-last-kernel interval. Prioritize linear-algebra
execution over cache-write elimination or host-scheduling changes. This is a
bounded instrumented window, not a full-workload throughput measurement.
Cache reads remain part of attention cost. Hardware-counter access is denied
by the Windows host, so compute versus memory-bandwidth saturation is not yet
established. See the experiment README and
`results/local_inference/profile_torch_early/kernel_summary.json` for evidence,
reproduction, and limitations.

After the user enabled Windows counter access and rebooted, hardware-counter
profiling succeeded. A six-launch capture across the three dominant GEMM kernel
names measured DRAM throughput 11.69–14.32% of peak, compute (SM) throughput
43.58–45.68%, and achieved occupancy 15.96–16.71%. Registers/shared memory limit
theoretical occupancy to 16.67%. This argues against saturated DRAM bandwidth
for these sampled kernels; kernel efficiency/latency hiding is a more promising
diagnostic direction. Neither low occupancy alone nor Nsight's suggested local
speedups proves a fix. Capture used serialization/replay and uncontrolled clocks
and caches, not benchmark conditions. Full evidence:
`results/local_inference/profile_counters_2048/details.txt` and
`results/local_inference/gemm_counters_2048.ncu-repz`.

Shape-aware follow-up matched all 2,448 GEMM/GEMV calls in a 16-step sample.
MLP gate/up (M,N,K = 2048,18432,2560) accounts for 30.87% of sampled GPU kernel
time, and MLP down (2048,2560,9216) for 16.34%: **47.21% combined**. Gated-delta
input projection adds 15.86%. The deeper six-launch counter capture identifies
math-pipe throttle as about 83% of the MLP between-instruction warp cycles,
with only 0.093–0.103 eligible warps/scheduler/active cycle and 6.09–7.02%
scheduler issue activity. DRAM throughput remains about 14%. This supports
testing BF16 kernel algorithms/tiling for those two exact MLP shapes first,
not optimizing cache writes or assuming a proportional gain from higher occupancy.
Tensor-pipeline activity is 43.54–46.14%; these kernel counters are **not MFU**.
No serving optimization has been applied. Evidence and limitations are in the
experiment README, `profile_shapes_2048/shape_summary.json`, and
`profile_deep_2048/details.txt` under the local inference result directory.

## Isolated BF16 kernel screen: retain the current path

One 256-call measurement window per candidate, using real layer-0 weights and
seeded synthetic BF16 activations, found no meaningful improvement from BLAS
preferences or eight Triton tiles. Gate/up default/best times were 1.98465/1.94998
ms (1.018x); down was 1.05308/1.02503 ms (1.027x). These small differences are
inconclusive under variable clocks and thermal throttling and do not clear the
predeclared 10% investigation threshold. No vLLM integration or dataset run.

Down default and non-strict cuBLASLt failed the synthetic maximum-error guard;
strict PyTorch and Triton passed and were closer to the FP32 reference, but differ
from default outputs. No judge-score parity is established. The runner initially
skipped the default down timing and crashed after timing the alternatives. Only
that missing reference was timed in a separate recovery process; validation
matched exactly, and no completed timing was repeated. All failures and original
outputs are preserved. See the experiment README and
`results/local_inference/gemm_bench_bf16/comparison.json` for the audited comparison.
