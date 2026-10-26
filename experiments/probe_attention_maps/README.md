# What changes in attention probes under soft targets

Follow-up to the [probe generalization study](../probe_generalization/README.md)
(run `full01`), NTP backbone only (user decision, 2026-10-26). On that
backbone, Kimi soft targets lift the attention probe family's OOD accuracy at
a 5% ID false-positive rate from 0.640 to 0.769. Of the rows that change
verdict across all five seeds, 711 flip to correct under soft targets, all of
them attacks, against 22 that flip the other way (21 benign). This study asks
what the probes read differently.

## Probe and decomposition

The attention probe is not causal self-attention. A per-token MLP (width 100)
feeds 10 heads, each with one learned query and one value vector. Each head's
softmax runs over all valid tokens of the row, so it yields one weight per
token, a_h(t), not a token-by-token map. Causality enters only through the
backbone: the layer-15 residual at token t has seen tokens up to t. The logit
is exactly

    logit = bias + Σ_h Σ_t a_h(t) v_h(t),

so each token's contribution is known. A seed gives both objectives the same
initial weights and data order, so hard-sK and soft-sK differ only in their
targets, and their heads correspond by index.

## Hypothesis and analyses (frozen before the GPU pass)

Hypothesis: soft targets change *where* the probe looks (attention mass moves
to different trajectory segments), not only *how strongly* it reads the same
tokens. The alternative, a uniform upward value shift on the same tokens, is
tested directly.

- **Probes.** `attention-{hard,soft}-s{0..4}` at epoch 8, the frozen ID pick of
  the attention family for both objectives. Per-probe threshold at 5% ID FPR.
- **Rows (`select.py`, CPU, saved logits).** Soft fixes (all 5 hard seeds wrong,
  all 5 soft right), hard fixes (the reverse), positives all 10 get right, and
  positives all 10 miss. Rank gain: seed-mean OOD percentile rank, soft minus
  hard. Examples for token maps: per source, the 2 soft fixes with the largest
  rank gain and the both-right positive with the median gain; the 4 hard fixes
  with the most negative gain. Picking by largest gain picks extremes; the
  aggregate statistics below cover every OOD row.
- **Segments (`segments.py`).** Each token is labelled by the trajectory header
  that precedes it: system, user, assistant text, tool call, tool output, the
  header itself (marker), or other.
- **Aggregates (`analyze.py`, all 6,395 OOD rows, every probe and head).**
  Attention mass and logit contribution per segment, attention entropy,
  attention centroid (relative position), mass in the last 10% of tokens.
  Per seed pair and head: attention overlap Σ_t min(a_hard, a_soft), and the
  cross terms Σ_t a_hard v_soft and Σ_t a_soft v_hard. These split the logit
  change into a part from moving attention and a part from changing values.
- **Gates.** The decomposition must match the bank's own forward on the first
  64 rows and every example, and the frozen post-hoc OOD logits (≤ 5e-2) on
  every row. Tokenization must reproduce each stored row's token count.
  Amended after the first launch stopped on row 1: the bank check had compared
  BF16 outputs (≤ 1e-2), but the full 110-probe bank and the 10-probe subset run
  different GEMM shapes and differed by 0.027. In FP32 they agreed to every
  printed digit, and the BF16 decomposition matched the frozen logits to 4e-6,
  so the bank check now runs in FP32 (≤ 1e-3) while the analysis stays BF16.

This is descriptive interpretability on frozen probes: no selection or
promotion follows from it.

## Compute

One Lambda GPU in us-west-3 attached to the `gleipnir-probes` filesystem
(user-authorized 2026-10-26). It reads the existing NTP OOD FP8 store; no
backbone forward is needed. `run_maps.sh` builds a light venv, stages the store
to local disk and runs `analyze.py`. Plots (`plot.py`) run locally from the
pulled `summary.npz` and example files.
