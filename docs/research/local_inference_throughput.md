# Local Gleipnir 4B inference campaign preparation

Prepared 2026-09-23. No optimization campaign has been launched.

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
will differ. Keep a separate longest-input startup canary. Confirm finalists
on all 3,012 rows; leave strict OOD outside optimization selection.

## Measurement and drift

Hypothesis: serving changes improve steady-state prefill throughput while
preserving the same judge's continuous scores and decisions. Begin with BF16
dynamic LoRA and a persistent vLLM engine; any lower-precision or merged model
is a separate intervention. Start conservatively on 16 GB: memory utilization
0.80, maximum sequences 2, batched-token budget 2,048, chunked prefill, and
32,768 context. These are proposed canary settings, not a verified fit or an
optimal configuration. Verify supported kernels and master/serving parity
including a nonzero adapter effect before scaling.

Repeat the baseline three times to establish its numerical variation. Warm up
before timing, clear prefix caches between passes, retain identical ordering,
and report initialization separately. Measure wall time, rows/s, prompt tokens/s,
peak memory, and all paired logit margins and scores. Compare absolute score
deltas, threshold flips at 0.5, ties, per-source and source-macro pAUROC@20,
AUROC, and Brier. Also inspect baseline-margin bins so saturated easy examples
do not hide drift near the decision boundary. Use paired, lineage-grouped
resampling for quality comparisons where grouping metadata is available.

Historical BF16 runs had nonzero repeat variation, so do not interpret every
changed score as optimization damage or select solely on aggregate AUROC.
Freeze numerical and quality tolerances after measuring baseline repeat noise
and before testing candidates. A 512-row screen can reveal paired drift but
cannot certify absence of small population-level quality loss. Stop on OOM,
truncation, missing/nonfinite scores, contract drift, or a failed serving-parity
canary. No candidate should be promoted before full-set confirmation.

## Runtime budget

For this one-token judge, prompt processing dominates. Time is prompt tokens
divided by measured prompt tokens/s; generated-token throughput is misleading.
The following are planning scenarios, **not measured RTX 4080 performance**:

| Sustained prompt tokens/s | 512-row estimate | Complete 3,012 rows |
| ---: | ---: | ---: |
| 5,000 | 19.1 min | 112.5 min |
| 10,000 | 9.6 min | 56.3 min |
| 20,000 | 4.8 min | 28.1 min |

Add model loading, compilation, warmup, and serialization. Three timed repeats
cost three times a single pass; reference and candidate both require passes.
For context, the historical H100 dynamic-LoRA pass processed 34,631,573 tokens
in 861.24 seconds (40.2k tokens/s), but that is not a workstation estimate.
Replace these scenarios with a stratified local pilot once inputs and weights
are staged, then calculate the estimate from its measured length mix.
