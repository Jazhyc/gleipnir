# Strict-filtered FP4/FA4 baseline

Establish a one-seed, one-epoch Qwen3.5-4B baseline for later quantization-aware
training using the selected native FP4 MLP/BF16 FlashQLA/varlen FA4 profile.
Use original unaugmented trajectories and unchanged Kimi soft targets, strict
teacher/source-label agreement at 0.5, seed 0, original zero-B rank-128/alpha-256
FP32 initialization, AdamW 5e-5, logical batch 32, 16,384 packed tokens and
29,696 context cap. Preparation must retain 7,836 of 8,688 rows (852 removed),
75,851,675 tokens and 245 updates. Preserve original inputs and filter provenance.

Select only the completed final epoch; no early stopping, held-out tuning or
promotion. Reuse the completed strict-filtered BF16/SDPA ID control, with no
new control training. The comparison changes both training arithmetic/attention
and serving precision, so it cannot isolate FP4. This establishes a baseline,
not a new QAT algorithm or evidence of seed robustness.

Use the shared monitoring campaign runner with `config.yaml`: `inventory`,
`prepare`, then `run`. Preserve FP32 masters on the existing NC2 network volume,
reuse shared compiler/kernel caches and checksum-bound selected-finite startup
validation, and record actual runtime/source/hardware bindings. Serving uses
the selected optimized vLLM 0.31 recipe and a single reconstructable BF16 merge
on ephemeral storage. New-adapter master/merge/optimized parity remains required;
no historical waiver applies to this adapter. Stop on drift, nonfinite/missing
gradients, OOM, failed parity or incomplete coverage; preserve failed receipts.

Evaluate the frozen 3,012 CoT-removed ID examples, reporting source-macro raw
normalized pAUROC@20, pooled/per-source AUROC, calibration, threshold diagnostics
and score ties. OOD and APPS are outside this baseline. Results live under
`results/filtered-fp4-fa4-baseline01/`, inputs under its matching `data/` tree,
and logs under `logs/runpod/filtered-fp4-fa4-baseline01/`. The Pod remains running.

No in-chat scheduler is available; active-turn checks verify startup and progress,
but detached process automation does not promise later agent follow-ups.
