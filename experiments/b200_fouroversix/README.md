# MLP-only Four Over Six training on B200

Hypothesis: native NVFP4 MLP forward/input-gradient GEMMs with adaptive 4/6
block scaling improve training throughput or activation memory relative to the
user-selected B200 NF4/BF16 QLoRA recipe, while permitting finite short training.
This is a bounded systems/learning pilot, not a quality-equivalence claim.

Use the existing B200 Pod `alzfug70g5237b`, leaving it running. No new capacity.
Keep the frozen 320-row stratified mixed-training cohort and Kimi soft targets,
Qwen3.5-4B revision, rank-128/alpha-256 FP32 adapters, AdamW 5e-5, clipping,
linear scheduler/3% warmup, 29,696 context, twelve linear checkpoints, SDPA,
FLA/fla-core 0.5.2, causal-conv1d 1.6.2.post1 and Triton 3.7.1. Compile the
established surrounding layers; keep native FP4 operations at an eager boundary.
Use adaptive physical batches of up to eight/16,384 padded tokens inside each
32-example logical update. Preserve the existing recipe decision and historical
failed gradient gates; this new diagnostic explicitly measures updates without
claiming NF4/FP4 gradient equivalence.

Intervention: replace only frozen HF decoder MLP gate/up/down bases with native
Four Over Six 1.0.5 (MIT), original checkpoint BF16 master weights, MSE 4/6
selection, 2D weight blocks, nearest forward and stochastic gradient rounding.
Use Triton quantization and CUTLASS NVFP4 GEMM with explicit backend selection;
prohibit simulated/reference fallback. Prepack frozen forward/transposed weights.
Higher-precision LoRA branches remain trainable; frozen bases need input
gradients but no base-weight gradients. Flatten all leading dimensions, because
upstream 1.0.5 linear backward asserts physical batch one. No upstream patch is
installed. The wrapper avoids retaining unused full-precision base inputs.

Controls: rerun optimized NF4/BF16 QLoRA, plus original-BF16 MLP bases with the
same NF4 attention weights/BF16 attention compute. Both selective controls load
original MLP weights rather than quantizing already NF4-rounded weights. Record
that attention storage remains NF4 as in the selected recipe; attention compute,
convolution, residuals, and normalization retain their existing precision.
This is LoRA training, not full-parameter FP4 training. Training uses no generation
KV cache. BF16 master bases and packed forward/transposed copies remain resident;
do not infer memory savings from operand bits alone.

Before model timing, execute native arbitrary-batch forward/backward canaries
against decoded FP4 matrix arithmetic (relative L2 <=0.02), a finite stochastic
backward check, and recorded native kernel provenance. Then run the original
global longest-32 materialized selection on the native candidate. Each timing
condition also checks eager/compiled same-weight loss (absolute 0.01 plus 1%
relative), longest-32 cohort backward, finite/nonzero adapter gradients, exact
restoration, and ten complete optimizer updates. Freeze a seed-0 permutation;
sort only within logical updates, match order and initial adapter hash across
conditions. The first scheduled LR may be zero; report it.

Stop on input checksum drift, missing kernels/backend, OOM, nonfinite tensors,
missing adapter gradients, restoration drift, compilation disagreement or runtime
failure. Preserve failed stages; do not reduce batches or change the recipe to
hide a failure. The opaque FP4 boundary can change Dynamo graph count; record
graphs and recompilations, and do not promote a graph-exploding implementation.
Report all step times and full loop time, peak allocated/reserved/process memory,
physical sizes/padding, losses and common eager training-probe losses before and
after using original BF16 MLP masters. No held-out selection or final-test access.
Use >=5% complete-loop gain as a systems interest gate; a promising candidate
needs a matched cached repeat and separate held-out validation before promotion.

Install the package without dependencies in an isolated persistent directory,
preserving uv.lock. PyPI source SHA256:
`51ab69c8c09e63d7575213f446f56bcbc9c36011ddca4c5d7b923710e1e6cb00`.
Record source hashes, source revision, inputs/selection hashes, kernel paths,
software, GPU and cache state in ignored contracts/results. Preserve FP32 adapter
artifacts on the volume; collect reports/contracts/logs locally.

```bash
source .cache-runtime.env
export FLA_DISABLE_BACKEND_DISPATCH=1
export PYTHONPATH="$PWD/.cache/kernels/fouroversix-1.0.5:$PYTHONPATH"
.venv/bin/python experiments/b200_fouroversix/run.py \
  --config experiments/b200_fouroversix/config.yaml
```

Inspect startup every 30–60 seconds, then progress/logs/GPU health every ten
minutes while the turn is active. No in-chat agent scheduler is available, so
after-turn monitoring is not promised. Collect and summarize completion.

The initial native arithmetic canary passed at all physical sizes, with about
0.00165 relative forward/input-gradient error against decoded FP4 arithmetic.
The compiled-base-boundary model preflight then failed its unchanged same-weight
loss gate: eager 1.352499 versus compiled 1.553454 (14.9% higher). No optimizer
update occurred. Preserve `results/b200_fouroversix/` and its logs as failed.

`eager_mlp_config.yaml` freezes a separate boundary diagnostic: keep complete
MLPs eager, retaining compiled surrounding decoder layers, checkpointing and
adaptive batching. Add an NF4/eager-MLP control to separate this boundary cost
from the precision effect; the original NF4 condition retains the selected
compiled recipe. BF16 and native MLP conditions share the eager MLP boundary.
Retain all original numerical gates and global-longest preflight. This is a
recorded implementation intervention, not a relaxed gate or silent fallback.

The full eager-MLP boundary also failed the unchanged model loss gate with exactly
the same eager/compiled values. Its native arithmetic canary passed, but no
long-context backward or optimizer step ran. At the user's request, stop after
this change: no ten-step controls or further variants were run. Both failures
and logs are preserved, and the existing B200 Pod remains running. See
[`b200_fouroversix_training`](../../docs/findings/b200_fouroversix_training.md)
for the completed finding. Do not promote this integration or infer training
speed, activation-memory savings or convergence from the isolated kernel tests.

Before resuming, read the
[handoff and proposed stability changes](../../docs/findings/b200_fouroversix_training.md#resume-checklist-and-stability-candidates).
It records the source/artifact/kernel state, the humans& recipe differences and
the order of bounded diagnostics. Those changes are proposals; the existing
training recipe and failed gates remain recorded as executed.
