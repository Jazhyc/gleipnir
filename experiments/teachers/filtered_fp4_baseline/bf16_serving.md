# BF16 serving follow-up

User authorization: evaluate the completed strict-filtered FP4/FA4 adapter with
regular BF16 serving, then diagnose and repair optimized-serving compilation.
The original failed optimized campaign remains frozen. This follow-up reuses
the exact FP32 master/export, master reference, ephemeral BF16 merge and passed
merged-reference receipt; it performs no training or merge. Its paired filtered
data retains the original removal provenance and token budget.

`bf16_serving.yaml` binds the original inputs and existing filtered SDPA ID
predictions. Prepare its Campaign contract directly on the already-filtered
inputs, link its adapter directory to the completed parent's adapter, and copy
the parent's merged artifact, merged parity and canary workload receipts. Invoke
only the shared campaign evaluate worker in the pinned serving runtime. Verify
the completed parent's bound artifact hashes before reuse and preserve a reuse
receipt. These reused stages do not claim new passes.

Require fresh master/merged/BF16-serving parity, native BF16 audits and complete
coverage of the frozen 3,012 ID examples. Report source-macro raw normalized
pAUROC@20, AUROC, calibration, threshold diagnostics, ties and exposure. No OOD,
checkpoint selection, promotion or parity waiver. Outputs are in
`results/filtered-fp4-fa4-bf16-serving01/`. Stop on drift, nonfinite outputs,
failed parity or incomplete coverage. Use the same ephemeral merge slot and
shared caches. Retire the BF16 scorer before compiler experiments.
