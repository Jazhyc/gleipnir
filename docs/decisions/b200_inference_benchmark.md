# Production inference development benchmark

Decision date: 2026-10-06. The user requests moving from training optimization
to production inference, with a small benchmark based on the existing training
systems split. Both interactive latency and batch throughput matter. Use the
already allocated US-NC-2 B200 pod `i243nsg10usytq`; the training worker has been
stopped. Do not provision capacity or change model precision in this baseline.

The completed ID pass scored 3,012 rows and 33,750,959 prompt tokens in
665.178 seconds (11 minutes 5 seconds), excluding startup and prompt preparation.
Its conservative engine memory fraction allowed coexistence with the retained
trainer, and its offline batches do not measure interactive HTTP latency.

Freeze the exact 320-row training systems selection, SHA256
`a27b8b3d702436839869fda2aa612ad61a4c3ef8aa9f56b4c57a4914b0a50e06`.
Create a nested 64-row quick workload from evenly spaced length ranks, including
both extremes, with seed-0 shuffled request order. Preserve source/prompt hashes
and report source composition. This is a systems development set, not a held-out
quality set; no final ID/OOD scores select an inference configuration.

Use the final FP4-trained adapter, checksum
`d13be8b249129269cec572e36e2006a955fa56dec5a015bd93f35bd876c62751`,
with the pinned BF16 Qwen3.5-4B base. Serving remains BF16 with dynamic rank-128
LoRA, FlashInfer GDN and a one-token binary decision. Historical SM120 inference
screens used different hardware, prompts and adapters; their MLP FP8 results
are candidate ideas, not the new B200 baseline or numerical acceptance.

The experiment README freezes the engine and scoring contract before launch.
Keep one localhost vLLM HTTP server across two passes each at client concurrency
1, 4 and 16. Disable prefix caching to prevent repeated-prompt speed inflation.
Measure response latency including server tokenization/queueing, request and
prompt-token throughput, length-bin tails and repeat score variation. This is
closed-loop load, with no production arrival-rate or SLO claim.

The pinned HTTP API cannot request explicit decision-token logprobs. Its binary
allowed-token mask plus processed top-two logprobs preserves the binary margin;
verify both finite token-ID entries and exact token counts, then compare against
the archived twenty-row master/serving reference and nonzero adapter effect.
No unchanged training preflight is repeated. Separate server readiness, parity
and four-length warmup from measured passes, and retain the first/repeat timings
so any residual first-use costs are visible.

Implement the protocol in `experiments/b200_inference_benchmark/` and reusable
selection/scoring/measurement contracts in `src/gleipnir/inference_benchmark.py`.
Record server PID/command, runtime/cache paths and executed sources; preserve
failed artifacts. Stop after the baseline suite and collect it before choosing
an optimization. Keep the serving process available for compatible future work.
Active-turn monitoring is available; no in-chat scheduler can promise later
agent follow-ups. No production endpoint is published.

## Storage expansion and preserved startup failures

The initial preparation rejects an optional trajectory checksum absent from
190 legacy records and their selection entries. Full source and selection
file checksums remain mandatory; accept the absent optional field only when
both records omit it. Preserve this attempt under `prepare_failure01`.

The first server reaches model loading, compilation and graph-memory profiling,
then fails because its subprocess PATH omits the installed `.venv/bin/ninja`.
Preserve the logs and executed sources under `startup_failure02`. Prepend the
virtualenv and CUDA binaries to PATH and check compiler availability before
launch; no package installation or arithmetic change is required. Eight focused
tests and Ruff pass. Restart with the same shared caches and frozen inputs.

The user requests expanding network volume `ixbh81vf9c` from 200 to 300 GB,
then explicitly permits doing so before the serving baseline finishes. The
update and fresh read-back both report 300 GB, `STANDARD`, `US-NC-2`, with the
same name `gleipnir-b200-workspace`. No volume, pod, dataset or cache is deleted.
The read-only audit reports 224,655,885,824 allocated file bytes across
`/workspace`: results use 132,503,187,968 and project caches 65,897,460,224.
Hugging Face weights account for 28,695,369,216 cache bytes and training caches
25,219,345,408. Mounted filesystem free-space figures describe the shared
filesystem rather than this volume's configured quota; use the Runpod API to
verify the capacity change. Preserve the resize and directory audit receipts.

The file audit explains the results accumulation: all 131 safetensors files are
`adapter_model.safetensors`, totalling 91,281,080,024 apparent bytes. Forty-two
causal masters and 42 final-checkpoint copies each occupy 29,294,468,688 bytes;
29 serving exports occupy 20,460,927,272 and 18 initial/resident-trial adapters
12,231,215,376. All 44 `.pt` files are named FP32 master snapshots, totalling
29,901,005,452 bytes. Their writers serialize adapter tensors, not AdamW
optimizer state. A 4B rank-128 FP32 master alone contains 169,869,312 elements
(about 680 MB); the latest epoch keeps a master, checkpoint copy and rebased
serving copy (about 2 GB together). File counts therefore do not represent
independent full training runs. Short systems screens and preserved copies
account for much of the growth. Allocated-byte directory totals and apparent
file-byte totals differ.

The user then explicitly authorizes deleting expendable short systems weights.
The path/size-bound manifest removes 127 files (86,300,534,628 apparent bytes),
verifies every planned file absent and all 48 protected weight files present,
and preserves logs, receipts, compiler caches, full quality/evaluation adapters
and the original frozen initialization. Collected local copies are retired too.
See the [retention decision](systems_training_adapter_retention.md) for the new
overwrite policy; no optimizer states or meaningful quality artifacts are removed.

## Completed serving baseline

All six quick passes complete after HTTP parity: adapter mean absolute score
difference 0.0038171181, correlation 0.9997699785, maximum adapter effect
0.7820802061. Base mean difference/correlation are 0.0056611087/0.9994300914.
The 64 rows contain 269,411 prompt tokens; full 320 contains 1,310,581.
Client concurrency 1/4/16 yields median 4.5312/8.3782/11.4255 requests/s and
19,074/35,269/48,096 prompt tokens/s. Median p50 response latency is
0.1536/0.4112/1.2552 seconds and p95 0.6394/0.8725/2.1425. Each statistic is
the median of two passes; full repeat and length-bin data remain in the results.

Across concurrency/repeat score arrays, mean/max per-row range is
0.0083116006/0.0621765009 and three rows cross 0.5. This measures baseline
variation, not a failed quantization candidate or held-out quality result.
Prefix cache is disabled and server logs report zero hits. Readiness wait takes
772.006 seconds; parity 18.235 and warmup 0.818, excluded from measured passes.
Earlier failed attempts and raw preparation are additional startup costs.
Collect all receipts, rendered inputs and executed sources; their hashes and
all six output memberships, finite scores and token counts are verified locally.
Server PID 71421 and engine PID 71522 remain alive/idle at localhost port 8000,
about 48.8 GB GPU memory, for compatible future inference trials. No further
optimization or full ID rerun is launched.

The user subsequently requests tokens/s alongside requests/s because request
lengths vary. Default comparisons lead with prompt tokens/s (19,074/35,269/48,096
at concurrency 1/4/16), retaining request latency and requests/s. The benchmark
already records this metric; no new measurement is needed. Distinguish prompt
throughput from the single generated decision token and hold workload/cache
policy fixed when comparing candidates. Record the preference in `AGENTS.md`.
