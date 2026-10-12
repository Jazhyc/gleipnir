# Quantization-aware recovery of the filtered 4B monitor

The [experiment](../../experiments/quantization_recovery/README.md) tests binary
same-model distillation from the completed strict-filtered BF16 monitor into a
student with merged-weight deployment quantization. The frozen optimized
baseline loses roughly 3.8 percentage points of ID source-macro raw normalized
pAUROC@20 against BF16 serving; see the
[baseline finding](filtered_fp4_baseline.md). This is one fixed recovery variant,
with no OOD selection or automatic promotion.

The first numerical preflight reproduces optimized-server training-source canary
scores with MAE 0.00573467 and correlation 0.99958407 (limits 0.02 and 0.99).
Its maximum score difference is 0.05868824, so this is aggregate canary alignment,
not exact per-example equivalence or held-out parity. The short two-sequence
packing probe has zero probability difference. Tested native FP4 projections
pass their quantized references. Receipts are under
`results/quantization-recovery01/attempt02/` on the NC2 network volume.

The subsequent longest-row backward probe runs out of memory at approximately
175 GiB allocated before any optimizer update. The correction shares gate/up
forward and decoded-input buffers and decodes packed FP4 directly to BF16,
avoiding full-size int64 gather tensors. It preserves rounding, original
contexts and logical batches. All 13 GPU decoder cases match the independent
oracle exactly. This allocation correction alone still exceeds memory at about
176 GiB. The subsequent correction retains packed MLP operands and reconstructs
gate/up activations within its custom backward, with an independent dense
forward/gradient control. The subsequent full preflight passes on the original
29,337-token longest row: all FP32 adapter gradients are present and finite,
unclipped norm 13.86179, peak allocated memory 137,556,206,080 bytes (128.11 GiB).
Alignment remains unchanged and the short packing probability difference is
zero. Preserve the original failed tracebacks and
receipts. An earlier bookkeeping failure came from archived `canary-N` display
IDs; joins now verify actual source IDs and rendered prompt hashes.

The fixed epoch completed all 245 updates over 7,836 rows in 3,955.95 seconds
(65.93 minutes), preserving the FP32 master with SHA256
`eb08e1371e72cde91243b4f0792270aa52ea97e855d9a9099a4b436f85659770`.
The original pipeline omitted the shared causal-to-multimodal adapter-key
conversion and stopped before merging. The export recovery uses the existing
checksum-tracked rebase helper, retaining FP32 values and the original master,
training contract and failure receipt. It resumes merge and the same frozen
evaluation without retraining; receipts live under the experiment's `recovery/`.

No recovery quality result is available yet. The forward uses native FP4 and
serving MXFP8 operations plus the FP8 quantized reference. The backward is an
approximate low-rank STE with quantized-QKV FA4 gradients. HF normalization and
BF16 FlashQLA recurrence retain differences from vLLM. This experiment does not
claim an exact serving derivative or that a short canary establishes ID quality.
