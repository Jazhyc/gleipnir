# Short QAD budgets

Hypothesis: 32/64 updates on representative subsets recover most of full QAD's
optimized-serving ID ranking, reducing its extra training cost. Both arms reset
to the completed strict-filtered BF16-trained FP32 adapter and consume its cached
same-prompt BF16 soft targets. No Kimi targets, teacher requests or control training.

Freeze nested 1,024/2,048-row subsets from the original 7,836 training examples.
Stratify by source, source label, teacher probability bands (0.1/0.4/0.6/0.9) and
global trajectory-length quartiles. Shuffle each stratum with seed zero; choose
the largest population-weighted quota deficit at each step, with deterministic
ties. The selection stream itself is the training order: the 32-arm examples and
logical batches are the prefix of the 64 arm. Record selected IDs, strata,
coverage, tokens and parent/derived checksums; preserve original teacher rows,
prompt hashes and labels. This selection does not change strict-filter membership.

Each arm uses logical batch 32, packing 16,384, context 29,696, KL T1, AdamW
1e-5, no weight decay, norm-1 clipping and a complete shortened linear schedule.
Ceil(3% warmup) gives one warmup update for 32 and two for 64. Reset adapter/RNG
and optimizer state; this changes data exposure and schedule versus full QAD and
does not isolate data quantity. Use original QAT arithmetic with redundant MLP
recomputation removed and frozen GDN work shared; BF16 FA4 attention surrogate
backward, decoded FP8/FP32 reference projections, native FP4/MXFP8 forward,
BF16 FlashQLA and FP32 masters. No packing-budget or experimental precision changes.

Reuse source-bound original trained-adapter startup and accepted structural
receipts for this unchanged validated recipe. Retain finite/missing-gradient
checks at every update. No repeated alignment/isolation/longest-row startup
probes. Final new-adapter master/optimized-serving parity/nonzero effect is mandatory.
Each fixed endpoint is merged with FP32 accumulation into the one owned ephemeral
BF16 slot, then gets two full frozen 3,012-row ID passes. Preserve both masters,
source archives, merge ownership history and compiler/kernel caches.

Compare cached full QAD mean macro raw normalized pAUROC@20 0.87608417 and
optimized un-QAD predictions, including pooled/per-source AUROC, calibration,
thresholds, ties, repeat variation and input tokens/s. Descriptive success is at
most 0.005 below full QAD; no ID tuning, adaptive extra epochs, OOD scoring or
automatic promotion. Stop on source/cache/hardware drift, invalid targets,
nonfinite/missing gradients, failed final parity, OOM or incomplete coverage;
preserve failures and stop the queue. Each stage cap is two hours.

`python -m experiments.qad_budget.launch` queues 32 then 64 on the existing NC2
B200. It freezes resolved arm configs after preparation, resets initialization
between arms and replaces only the checked owned merge. Outputs are under
`results/qad-budget-sweep01/` and `results/qad-budget{32,64}-01/`; derived training
views under matching `data/` directories. Logs use the same names under
`logs/runpod/`. No in-chat scheduler is available; detached orchestration does not
promise later agent wakeups. Monitor startup until real updates advance.
