# MXFP8 prefill as the B200 inference optimization reference

Decision date: 2026-10-06. The user accepts the completed MXFP8 result and asks
to make it the new reference point for optimization. This supersedes the
[FP4-projection/BF16-attention reference](b200_fp4_gdn_inference_baseline.md)
for new comparisons; preserve all historical controls and failures.

Use all 64 native FROST FP4 MLP projections and 48 large FP4 GDN projections,
plus the validated adapted cuDNN MXFP8 D256 causal/GQA prefill kernel. Q/K use
rowwise blocks and V columnwise blocks, with E4M3 payloads and E8M0 scales.
Use the checksum-named hardware-exp2 variant, not the unmodified stock kernel.
Keep small gates, convolution/recurrence, persistent KV cache and decode BF16,
with FP32 gates/state. Keep the FP32 master adapter, ephemeral merged BF16
checkpoint and shared compiler/kernel caches.

## Evidence and explicit acceptance

The corrected `fp4_gdn_cudnn_mxfp8_02` completes all fourteen serving passes.
`mxfp8_confirmation02` supplies five further warm c128 passes, median
173,938 input tokens/s (+4.83% versus the preceding reference's 165,927).
Source-macro AUROC changes +1.68 percentage points and pooled AUROC −0.37
points on the frozen training-seen development cohort. Preserve calibration,
threshold flips, timing variation and source-level results. See the
[finding and current profile](../findings/b200_mxfp8_serving.md).

`quality_acceptance.json` separately records `user_accepted_finite`. The native
strict FP32-reference check remains failed at 5.336% against the unchanged 5%
ceiling. Strict master-score parity remains failed too. Baseline-relative score
agreement passes and all seven corrected producer/arithmetic/replay cases pass.
Do not change those historical classifications or widen any ceiling. The first
incorrect cache-page/token-offset integration and its invalidation records remain
preserved. Finite outputs, matching identities/prompts/tokens and nonzero adapter
effect remain mandatory. This acceptance covers the current adapter and pinned
arithmetic, not new precision drift or other adapters automatically.

## Reference resolution and worker reuse

`experiments/b200_inference_benchmark/baseline.json` binds the result, executed
recipe, input manifest, full sweep, five-pass confirmation, native validation and
explicit acceptance by checksum. The preceding selection is archived under
`baselines/frost_fp4_mlp_gdn_bf16_attention.json`. New candidates use
`baseline: selected` and `high_reference: selected`; the latter also verifies
the selected full-sweep hash. Explicit historical references are unchanged.
Use 173,938 input tokens/s for warmed c128 confirmation comparisons, retaining
the full-sweep results for per-concurrency comparisons.

The attention/GDN runner defaults to `fp4_gdn_cudnn_mxfp8.json`. Its comparison
selector now follows the selected reference; the validated executed copy retains
the historical preceding comparison. This metadata change does not change the
kernel arithmetic or require restarting the resident worker.
Reuse permits a changed `high_reference` in the client comparison metadata,
while still checking all loaded-kernel hashes, precision and capacity arguments.

The current API **84940** / engine **85056** stays warm on the existing NC2 B200,
port 8010. Reuse it for matched requests and profiling; stop it before changing
active kernels/settings. No control replay, new capacity or public deployment is
required by this selection. Continue reporting AUROC for future kernel changes.
