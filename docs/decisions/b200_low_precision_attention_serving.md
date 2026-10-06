# B200 serving precision after FROST baseline selection

Date: 2026-10-06. The user selects FROST FP4 MLPs as the inference optimization
baseline, prioritizing higher-concurrency throughput, and asks about lower
precision for GDN/linear attention and full attention. This is a source-level
feasibility assessment; no additional precision trial, package upgrade or
server restart is performed.

The selected baseline is `results/b200_frost_inference/frost02`, bound in
`experiments/b200_inference_benchmark/baseline.json`. Its merged model has FP4
MLPs, BF16 other projections, FlashInfer GDN and FlashInfer/TRTLLM full attention.
FlashQLA and BF16 FA4 are training backends, not the active serving backends.
The existing API PID 74857 returns health 200 and remains resident with engine
PID 75080. Use saved FROST timings and six prediction arrays as the comparison;
retain explicit historical BF16/FP8 comparisons without rerunning them.

## Full attention: native FP8 is the first candidate

The installed stack is vLLM 0.24.0/FlashInfer 0.6.12. Its FlashInfer backend
supports FP8 E4M3/E5M2 KV caches and, when TRTLLM is selected and query
quantization is enabled, selects the query dtype to match the FP8 cache.
There is also a BF16-query/dequantization fallback, so an FP8 cache flag alone
does not prove FP8 attention computation. Audit the actual query/KV dtypes and
prefill kernel before describing a trial as FP8 compute. The source has an
NVFP4 KV-cache path using FP8 queries; that is a separate, more aggressive
candidate, not proof of all-FP4 attention arithmetic.

Evidence: [pinned vLLM backend](https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/v1/attention/backends/flashinfer.py),
[FlashInfer causal paged prefill API](https://docs.flashinfer.ai/generated/flashinfer.prefill.trtllm_batch_context_with_kv_cache.html).
Installed file hashes and dtype evidence are saved in
`results/b200_low_precision_attention_serving/installed_source_audit.json`.
The installed path must still prove causal D256, 16-query/4-KV GQA, ragged
request lengths, hybrid cache layout and CUDA-graph compatibility. This source
audit does not claim a successful kernel launch or numerical validation.

Recommend FP8 Q/K/V full attention while keeping FROST MLPs and BF16 GDN fixed.
Use recorded, appropriate quantization scales and count their conversion costs;
never assume unit scales or reduced cache memory guarantee speed or parity.
Our one-token monitor workload is predominantly prefill, so the experiment
must measure complete input-token throughput, not infer benefit from decode
benchmarks or cache capacity. Hold concurrency/token/memory budgets fixed in
the first screen; capacity/concurrency tuning is a separate intervention.

## GDN: separate projection GEMMs from the recurrence

The installed vLLM FlashInfer wrapper explicitly converts GDN state, forget
gate and update gate to FP32. Its Mamba state dtype configuration exposes
auto/FP32/FP16/BF16, not FP8. The current pinned path has no configuration-only
FP8 recurrence option comparable to full attention. Do not confuse quantized
QKV/output projection weights with low-precision delta-rule GEMMs or state.

Upstream [FlashInfer GDN prefill documentation](https://docs.flashinfer.ai/generated/flashinfer.gdn_prefill.chunk_gated_delta_rule.html)
now lists FP8 initial/output state and checkpoint storage, while its Q/K/V
paths retain FP16/BF16 inputs. This is newer than our installed 0.6.12 and
does not establish FP8 tensor-core recurrence, a compatible pinned vLLM path
or an end-to-end benefit. In the installed wrapper, casting state back to FP32
would also have to be addressed. State compression is most relevant to cache
capacity and continued generation; it is not automatically a prefill gain.

Recommend screening native FP8 **GDN projection GEMMs** separately, retaining
BF16 convolution/recurrence operands and FP32 gates/state. Consider FP4
projections after their own score checks. A true low-precision recurrence
would require kernel work on chunk matrix operations and triangular solve,
normalization, scale propagation and state accumulation, rather than changing
one dtype. Keep it below native FP8 full attention in priority. BF16 FlashQLA
versus FlashInfer serving could also merit a matched forward-only comparison;
there is no evidence that the currently selected BF16 kernel is optimal.

Future arithmetic changes need adapter-specific canary results and the same
64-row speed/AUROC screen against the selected FROST baseline. Report pooled,
per-source and dual-label source-macro deltas, score drift, thresholds and
repeat variation. Preserve finite failed parity explicitly as diagnostic and
stop on structural/native/nonfinite failures. The user has not requested a
new trial in this turn; keep the baseline warm and record these next candidates.
