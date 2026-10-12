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

## Completed ID comparison

Both final-checkpoint passes cover all 3,012 frozen ID rows. Saved metrics were
recomputed exactly from each prediction file; receipts and raw predictions are
under `results/quantization-recovery01/`. All three new-adapter serving canaries
pass, with master-score MAE 0.00387763, 0.00387763 and 0.00515554, correlations
above 0.99977 and a nonzero adapter effect. The original export failure and
failed startup remain archived; `recovery/status.json` records successful completion.

| Model / serving | Macro pAUROC@20 | Pooled AUROC | Pooled Brier | Pooled ECE | Input tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original monitor, BF16 | 0.887045 | 0.956521 | 0.082633 | 0.047501 | 112,743 |
| Original monitor, optimized, two passes | 0.847754–0.850308 | 0.940109–0.941193 | 0.097521–0.098278 | 0.054901–0.058461 | 177,828–179,427 |
| QAD monitor, optimized, two passes | 0.875735–0.876433 | 0.949218–0.949311 | 0.087650–0.087798 | 0.039925–0.040775 | 172,576–174,524 |

Mean macro pAUROC improves by 2.7053 percentage points, recovering 71.17% of
the original 3.8014-point BF16/optimized gap, with 1.0960 points remaining.
The improvement is concentrated in Gloom: pAUROC 0.793433–0.793867 versus
0.738401–0.742780 before recovery and 0.818870 in BF16. STRIDE is nearly
unchanged at 0.958037–0.958999 versus 0.957107–0.957835, above the BF16 0.955219.
At the fixed 0.5 threshold, pooled recall rises to 0.873134–0.876866 from
0.838308–0.840796, while false-positive rate rises to 0.106125–0.107550 from
0.081197–0.081909. Preserve this operating-point tradeoff alongside ranking gains.

Repeat score MAE falls from 0.01372785 to 0.01191295, correlation rises from
0.99605030 to 0.99681882, and threshold flips fall from 41 to 34. The largest
score difference remains substantial, 0.360363 versus 0.353518 before recovery.
This is modest repeatability improvement, not stable per-example equivalence.
Serving throughput is roughly 1.54 times the cached BF16 baseline and slightly
below the earlier optimized passes. Timing variability is not independently isolated.

This one-seed, fixed-epoch ID result supports partial recovery of deployment
precision loss. No OOD evaluation or default promotion was performed. It does
not isolate an optimal distillation teacher or establish transfer.

The forward uses native FP4 and
serving MXFP8 operations plus the FP8 quantized reference. The backward is an
approximate low-rank STE with quantized-QKV FA4 gradients. HF normalization and
BF16 FlashQLA recurrence retain differences from vLLM. This experiment does not
claim an exact serving derivative or that a short canary establishes ID quality.
