# Merged-weight QAT training optimization

Hypothesis: shared frozen GDN projections and removal of unused MLP backward output
recomputation reduce complete-update time while retaining the completed QAD
quantizers, FP32 master adapters and approximate STE objective. Keep the historical
implementation and its checksum-bound validation receipts unchanged.

Use the authorized NC2 B200 and the pinned Torch 2.11 training environment with
shared persistent caches. Load one resident model from the original strict-filtered
baseline adapter, replay seed-zero first twenty logical batches (32 examples,
16384-token packing budget), and reset identical FP32 masters, RNG and fresh AdamW
state before every trial. Use the original 245-update learning-rate schedule.
Baseline, optimized and optimized-repeat are twenty complete updates each. Compare
updates 11–20; report warmup separately and retain every sample. Archive source
hashes, losses, gradient norms, physical partitions, master digests and cache counts.
Overwrite only `results/systems_training_scratch/` weights. Keep the worker alive
after the queue; its PID and requests live under the experiment output.

Before timing, reuse the checksum-bound successful original startup preflight for
unchanged kernels and verify hardware/runtime/source identities. New native FP8
operators must be within 0.01 relative L2 of the unscaled FP32 quantized reference.
Run a baseline repeat and matched first-logical-batch loss/gradient diagnostic. Require score MAE
<=0.002, maximum difference <=0.01, KL absolute difference <=0.0001, and adapter
gradient relative L2 <=0.01 to call the candidate numerically accepted. Missing or
nonfinite gradients stop execution; failed numerical gates retain diagnostic
receipts and do not enable the optimized path. Select the systems candidate only
if both repeats are >=2% faster than baseline and introduce no new FP4 plans in
measured updates. These screens cannot establish full-corpus or held-out quality.
Stop on OOM, checksum/reset drift, nonfinite values or a 30-minute active-work cap.

Targets remain the completed same-model BF16 cache for the comparison. The loader
also validates the existing strict-filtered Kimi K3 cache using source identities,
source labels and separate privileged-teacher/student prompt hashes. No new teacher
calls or Kimi training campaign are authorized by this screen. No ID/OOD evaluation
or model promotion occurs. This session has no in-chat scheduler; monitor actively.

The initial `qat-training-optimization01` native-FP8-plus-MLP candidate passed
isolated GEMM checks but failed the original full-model gates (score MAE 0.005474,
maximum 0.031552, KL difference 0.001007, adapter-gradient relative L2 0.813334).
Retain its exact source archive and failed receipts. The subsequent
`qat-training-optimization02` disables native FP8 and shares only frozen GDN
QKV/Z contraction/input packing, preserving separate STE backward calls. The
existing FP32 FP8 reference remains selected. Numerical limits are unchanged.

Launch on the existing remote checkout:

```bash
/tmp/gleipnir-serving-runtime/venv/bin/python -m experiments.qat_training_optimization.launch
```

The reusable context applies after `quantization_recovery.install(model, root)`:

```python
from gleipnir.training.qat_optimization import optimized_path

with optimized_path(model, fp8=False, mlp=True, gdn=True):
    # Run the normal packed binary-KL training loop.
    ...
```

`gleipnir.training.distillation_targets.load_binary_targets` accepts the frozen
Kimi cache with `teacher_model="moonshotai/kimi-k3"` and
`prompt_contract="privileged"`; record its returned binding receipt. The config
contains both exact cache checksums. Actual measurements and qualifications live
in the [finding](../../docs/findings/qat_training_optimization.md).
