# Selected merged-weight QAT training recipe

The user selects the original QAT arithmetic with redundant operations removed
on 2026-10-13. This selection applies to future merged-weight QAT runs; it does
not change ordinary training or frozen historical campaigns.

After `quantization_recovery.install(model, root)`, use
`gleipnir.training.qat_reference.attention_runtime(model)`. It selects native
NVFP4 MLP/GDN forward, the decoded quantized FP8 projection reference with FP32
contractions, serving-matched MXFP8 attention forward and quantized-QKV BF16 FA4
surrogate attention backward. Keep BF16 low-rank STE contractions, FP32 master
adapters/gradients and the selected BF16 FlashQLA recurrence. The context removes
the unused MLP down-output recomputation in backward and shares frozen GDN QKV/Z
packing/contractions, with separate STE backward calls.

The [structural screen](../findings/qat_training_optimization.md) reproduces the
tested twenty-update losses, norms and final adapter tensor digest exactly.
It establishes arithmetic preservation on that screen, not a new full-epoch
quality result or a guaranteed wall-time gain. Preserve its validation/source
bindings; changed kernels or execution envelopes require targeted checks.

Native FP8 projection GEMMs and native MXFP8 backward remain experimental after
their [projection](../findings/native_fp8_qat.md) and
[backward](../findings/native_mxfp8_qat.md) ID regressions. No new training,
evaluation or model promotion is part of this default selection.
