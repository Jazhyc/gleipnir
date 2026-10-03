# Behavior-grounded 4B monitor: campaign startup

2026-10-03. Training and evaluation are in progress; there are no held-out results
yet. The approved design is in
[the experiment README](../../experiments/monitoring_injection_disentanglement/README.md).
ID uses the conservative training instruction once over all 3,012 frozen rows.
Only the injection benchmark and honest controls receive the three-prompt grid.
The requested benchmark is `mkenney2/monitor-injection-evals`, branch
`feat/gleipnir-vllm-grid`. The Runpod checkout is at `29b31d9`; the local checkout
at `b55c201` adds reports only. Both frozen input checksums match.

All 8,688 training rows and the original teacher targets are retained. The pinned
token audit totals 86,900,609 input tokens, with maximum 29,692 under the unchanged
29,696-token cap and no truncation. The selected packed BF16/FlashQLA recipe,
FP32 adapters, one epoch and seed zero remain unchanged. Startup diagnostics are
explicitly reused from the prior validated recipe, with mandatory gradient checks.

The first launcher mistakenly created a campaign-specific cold compiler cache.
It reached ten updates with finite reported loss/gradient norms, but compilation
caused long GPU-idle intervals. Preserve this attempt's logs and execution receipts
under `attempts/cold_cache/`. The runtime now shares the populated persistent cache;
missing entries from the cold attempt were merged without overwriting old entries.
The restarted run begins from the same initial adapter for a complete epoch.

The often-cited 5.1452 seconds/update comes from the 320-row mixed systems cohort,
which processed 1,314,331 tokens over ten updates (131,433 tokens/update). This
campaign averages 319,488 input tokens per logical update, about 2.43 times that
workload. The previous full 8,688-row regular-prompt run measured 12.2587 seconds
per update after its first ten updates (median 12.3123). Its metadata checksum is
the frozen startup-reference checksum in this campaign's configuration. Compare
the new run with that matched workload; a token ratio alone does not predict
performance, especially with long independent full-attention segments. The
[packing finding](bf16_sequence_packing.md) retains the original five-second result.

The restarted worker's environment verifies that Inductor, Triton, TileLang and
TVM use the populated `student_injection_awareness` cache. Its first update took
about 266 seconds, versus 901 seconds for the initial cold attempt. Progress
timestamps for updates 2–37 imply a mean and median of 13.0 seconds/update,
excluding the first update; the most recent twenty updates averaged 13.2 seconds.
A bounded one-minute measurement (twelve samples, five seconds apart) averaged
99.4% GPU utilization. The measurement receipt is
`results/monitoring_injection_disentanglement/throughput_snapshot.json`.
These are startup observations, not a whole-epoch utilization average or held-out
quality result. Training is ongoing.
No new recipe benchmark, data exclusion, teacher relabeling or evaluation-based
checkpoint selection follows from this startup correction.
