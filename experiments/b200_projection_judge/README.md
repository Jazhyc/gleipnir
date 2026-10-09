# APPS-fitted full projection transferred to JudgeDeceiver

User-authorized follow-up to the completed injection-direction experiment.
Hypothesis: the same APPS-fitted layer-20 direction also controls injection
avoidance in A/B quality judging. Apply the saved unit direction at all 32
layers/all tokens, beta 1 and zero centers, unchanged. Fit nothing on
JudgeDeceiver; no clamp, layer/strength sweep or promotion.

Reuse the exact original 4,188 native A/B prompts, both orders, 252 pairs and
six held-out query groups: 504 clean, 1,842 preferred-injected and 1,842
disfavored-injected. Labels are preferences, not harmfulness. Original source,
construction-label provenance, grouped split, dependent suffix/order limitations
and unresolved upstream license are defined in
`experiments/judge_injection_continuation/README.md`. The current SDPA adapter
did not train on this dataset. Historical FA4/other-adapter scores are context;
the control is fresh unedited scoring on the identical current SDPA backbone.

Keep the user-authorized BF16 eager Lens research recipe and merged checkpoint
on the existing NC2 B200. Extend full-vocabulary readout with A/B embedding rows
32/33, probability mass and an independent FP32 selected-row head reference,
using the actual fused final normalization's hidden vector. This leaves the
resident 0/1 cached head and all backbone arithmetic intact. Replace the old
engine sequentially, retaining its receipts/caches. Reuse unchanged adapter/
merge parity, and require fresh existing 0/1 master/native/functional gates,
no-op reproduction against the previous eager canary (MAE <=.005,
correlation >=.995), tokenizer A/B identity and readout parity on the original
30 training-derived canaries (MAE <=.020, correlation >=.99 against the FP32
selected-row head). The latter checks the readout, not a new whole-backbone
FP32-master A/B comparison; preserve that qualification. Require beta-zero
exactness, a following plain request restored exactly and zero request state.

Freeze workloads, input/source hashes, server identity and projection receipt
before full evaluation. Score unedited and projected A/B p(B) at fixed .5,
with ties choosing B. Report condition/source/query accuracy, AUROC, raw p20,
calibration/ties, paired clean/injection effects and backend decision changes,
plus p(A or B). Capture full-vocabulary readouts for every row, concurrency 32,
128-row batches, prefix cache off; report input tokens/s, requests/s and latency
as single-pass diagnostic costs. No generation or format-robustness claim.
The six-query grouping limits uncertainty/generalization; no threshold tuning.

Stop on source/input/resident drift, failed startup/readout gates, truncation,
nonfinite outputs, missing coverage/state cleanup, OOM or completion. Resume
only exact checksum-bound completed batches; retain failures. Keep the passing
monitor engine warm. No capacity changes, retraining, fitting or local merged-
weight transfer. Active-turn monitoring only; no heartbeat tool is available.

Run `python -m experiments.b200_projection_judge.run` inside the pinned serving
runtime after startup `sdpa03_judge_projection`. Config owns frozen inputs;
artifacts use `results/b200_projection_judge/sdpa_judge_projection01/`.
