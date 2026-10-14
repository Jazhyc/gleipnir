# B200 FP8 default single-request contexts to 1M

2026-10-15. The current FP8 serving default completes exact 1,048,576-token
inputs at concurrency 1 on one B200, in a median 43.08 s (24,340 input
tokens/s). Through 256K it closely tracks the superseded 0.24 FP4 stack. Beyond
64K, full attention dominates. At long contexts the GPU hits its 1,000 W power
cap and the SM clock drops, so attention-heavy requests run below the nominal
clock. See [the experiment contract](../../experiments/b200_context_million/README.md).

## Node and environment

User-provisioned Runpod Pod `cvjantiismcaod`, EU-RO-1, $7.99/hour, network
volume `cb7bxu71ug` at `/workspace`. B200 `GPU-3d8d6bde-5512-8b51-90a1-09e467c5bd50`,
VBIOS 97.00.D9.00.35, PCIe 5 x16, ECC on, MIG off, 1,000 W power limit, max SM
clock 1,965 MHz. **Driver 580.126.20 / CUDA 13.0**; earlier Pods recorded
580.178.04 (EU, 2026-10-07) and 595.91.07 (NC2/CA2). Host: 2x AMD EPYC 9555
(224 threads), 2.2 TiB RAM, kernel 6.8.0-107, Ubuntu 24.04.4. Runtime: vLLM
0.31.0, Torch 2.13.0, Triton 3.7.1, Transformers 5.14.1, FlashInfer
0.7.0.post1, cuDNN 9.20.0.48 / frontend 1.31.0, CUTLASS DSL 4.8.0, TVM FFI 0.1.11.
No environment drift was observed between the start and end snapshots.

The fresh volume lacked the serving stack. The checksum-bound 2026-10-07 EU
bundle restored the native overlays and Gigatoken. The locked
`.venv-vllm031` was installed and staged. The selected merged checkpoint was
rebuilt from the FP32 master, and every file matched the accepted hashes. The
prior EU checkout is preserved as `/workspace/gleipnir-eu-20261004`.
Receipts are under `results/b200_provisioning/20261015_eu_long_context/`.

## Validation

The MXFP8 descriptor/producer bound alone is raised to 1M. Short and
128-sequence outputs stay bitwise identical. Full 32K query chunks against
64K–1M histories, with pages beyond 2^31 cache elements, are finite. Sampled
quantized-oracle error is 0.164–0.168%. Strict BF16 relative L2 is
0.0532–0.0560, failing the 0.05 diagnostic gate as at 256K before. This is
the known MXFP8 deviation, not a new failure mode.

In the native-RoPE phase, the 20-row canary reproduces the archived FP8
default exactly (MAE 0, correlation 1, adapter effect 0.845). This validates
the restored node. Three full320 c128 passes against the six archived FP8
repeats give score MAE 0.0015, one threshold flip, and pooled/macro AUROC
deltas of −0.025/+0.023 pp. This short-input batch diagnostic is not
long-context quality.

Beyond 262,144 positions, the YaRN phase uses Qwen's static factor-4 YaRN with
mRoPE sections retained. vLLM 0.31 requires the scaled 1,048,576
`max_position_embeddings` in the override. Static YaRN also changes short
scores: the canary gives MAE 0.028 and correlation 0.993 against the default.
This was recorded as a diagnostic, as planned. Timings are valid. Scores
beyond 256K are unvalidated monitor outputs, and long-context quality is
unmeasured.

## Measurements

Five timed requests per length after one excluded warmup. Prompts at 8K–256K
are byte-identical to the archived `long02` workload. Startup (164 s native
with cold 0.31 caches on this node, 72 s YaRN) is excluded.

| Context | Server | Median latency | Range | Input tokens/s | Median / min SM clock |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1K | native | 35.47 ms | 34.9–35.8 | 28,833 | 1965 / 1965 |
| 2K | native | 35.04 ms | 34.7–36.9 | 58,369 | 1965 / 1965 |
| 4K | native | 37.93 ms | 37.3–41.5 | 107,878 | 1965 / 1965 |
| 8K | native | 55.82 ms | 53.9–59.8 | 146,564 | 1965 / 1965 |
| 16K | native | 98.66 ms | 98.3–108.7 | 165,958 | 1965 / 1897 |
| 32K | native | 207.49 ms | 206.4–208.4 | 157,886 | 1905 / 1777 |
| 64K | native | 484.47 ms | 480.7–487.5 | 135,255 | 1815 / 1695 |
| 128K | native | 1,256.5 ms | 1,253.8–1,261.1 | 104,312 | 1717 / 1567 |
| 256K | native | 3,648.0 ms | 3,641.5–3,651.5 | 71,857 | 1620 / 1477 |
| 512K | YaRN | 11.70 s | 11.66–11.89 | 44,801 | 1552 / 1155 |
| 1M | YaRN | 43.08 s | 42.97–43.21 | 24,340 | 1462 / 997 |

Overlap points on the YaRN server: 32K 196.86 ms and 256K 3,542.95 ms (0.95x
and 0.97x native). RoPE is a table lookup, so this reflects server-instance or
thermal state, not a scaling effect. Against `long02`, the native phase is
1.00–1.08x the old latency (8K +8.4%, 32K +3.3%, 256K +1.3%). Stack and node
both changed, so no cause is attributed.

**Fixed floor.** Latency is flat at about 35 ms from 1K to 4K. Short C1
requests are dominated by per-request overhead, not compute.

**Chunk cost model.** A relative-error weighted fit of
`latency = n*a + b*n(n+1)/2` over whole 32K chunks (n = 1…32) gives
a = 134.6 ms per chunk and b = 71.8 ms per 32K×32K attention block (8 layers).
It is within 2.0% at every point from 32K to 1M. The attention term is 35% of
latency at 32K, 71% at 256K and 90% at 1M. The per-chunk constant matches the
batched short-input saturation rate of roughly 245K tokens/s. b implies about
2.0 PFLOP/s for causal MXFP8 attention, averaged over the falling clock.

**Power.** Event reason `0x4` (software power cap) appears from 16K onward.
Peak board power reaches 990–1,010 W from 128K, and the median SM clock falls
from 1,965 to 1,462 MHz at 1M, with a 997 MHz minimum. Compute-bound long
attention is power-limited on this node.

**Memory.** Peak incremental PyTorch allocation is 2.43 GiB from 32K through
256K, 2.75 GiB at 512K and 3.79 GiB at 1M. Absolute peak allocation is
162.13 GiB and reserved memory 174.69 GiB at 1M. These include the
preallocated KV pool and a 4M-position YaRN rotary table. Telemetry observes
32K query chunks against every history up to 1,048,576. This shows complete
C1 processing; it does not establish multi-request 1M capacity.

## Artifacts and closure

`results/b200_context_million/million01/` holds the native receipt, per-phase
commands, canaries, RoPE audits, quality diagnostic, raw repeats, 250 ms GPU
samples, environment snapshots (raw `nvidia-smi -q -x`/`lscpu`), and the
summarized `scaling.{json,csv,png,svg}`. Server logs are under
`logs/runpod/b200_context_million/`. Both servers were retired by identity
check. The GPU is idle (0 MiB) and the Pod remains running. No capacity
lifecycle operation was performed.
