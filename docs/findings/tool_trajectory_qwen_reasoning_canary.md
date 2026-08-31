# Qwen3.5 reasoning-mode tool-trajectory canary

Status: completed 2026-08-30. This is a 100-row in-distribution source canary,
not the paper's final STRIDE test or Gloom-Exfiltration evaluation.

## Question and frozen protocol

The canary tested whether reasoning before the binary monitoring boundary helps
the unadapted `Qwen/Qwen3.5-9B` express better-ranked or better-calibrated
uncertainty than immediate decision logits. It used model/tokenizer revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, BF16 vLLM 0.24.0, and one H100
80GB. The frozen config was commit
`f6d76b73cd1ca5427f13ad1774463d2bedc33a56`; its SHA-256 was
`3cc214a1085cdf0677a45982c9e4f7de931dea4f8927fa51e03f120961be722e`.

The ignored input artifact had SHA-256
`43106194e2e355aab4292afdcb02d70fc654f227071282eec52ee17cea3ae3b1`.
It reused the deterministic teacher-canary selection: 25 rows from each
source-label stratum of `gemini_synthetic` and `unified_dataset`, from
`Syghmon/blackboxmonitorsMATS` revision
`ccb77e37589baaee1f0e16ac9ab89ceac8f6cc8a`. These accessible synthetic
releases are earlier source data, not the final paper manifests. All rows were
at most 28,000 source tokens.

The three conditions were:

1. immediate non-thinking logits after the compact student prompt's literal
   `Prediction:` boundary;
2. visible deliberation with native thinking disabled; and
3. the same deliberation prompt and sampling settings with native thinking
   enabled.

Both reasoning arms used temperature 1.0, top-p 0.95, top-k 20, presence
penalty 1.5, a 128-token minimum, a 16,384-token maximum, and seed 20260830.
The harness removed the generated terminal decision digit and made a constrained
second pass for the literal `0|1` logits at that exact boundary.

## Results

| Condition | Macro AUROC | Pooled AUROC | Macro Brier | Balanced accuracy | Mean entropy (bits) |
|---|---:|---:|---:|---:|---:|
| Direct logits | **0.8672** | **0.8378** | **0.2008** | **0.7500** | **0.4895** |
| Visible reasoning | 0.7432 | 0.7484 | 0.2977 | 0.7000 | 0.0083 |
| Native thinking | 0.8344 | 0.8036 | 0.2466 | **0.7500** | 0.0087 |

Per-source AUROC was `0.9360 / 0.7984` for direct logits,
`0.9152 / 0.5712` for visible reasoning, and `0.9168 / 0.7520` for native
thinking, where each pair is Gemini synthetic / unified dataset.

A 5,000-resample paired bootstrap with seed 20260830 preserved all four
source-label stratum sizes. Intervals below are percentile 95% intervals for
candidate minus baseline:

| Comparison | Macro AUROC delta (95% interval) | Macro Brier delta (95% interval) |
|---|---:|---:|
| Direct to visible | -0.1240 `[-0.2008, -0.0528]` | +0.0968 `[+0.0411, +0.1583]` |
| Direct to native thinking | -0.0328 `[-0.1080, +0.0456]` | +0.0458 `[-0.0056, +0.0986]` |
| Visible to native thinking | +0.0912 `[-0.0136, +0.1976]` | -0.0511 `[-0.1154, +0.0104]` |

Visible reasoning generated 31,364 tokens in total (mean 314, maximum 617).
Native thinking generated 178,106 tokens (mean 1,781, maximum 5,747). All 100
visible outputs and all 100 native-thinking outputs had a parseable terminal
decision; all native-thinking outputs completed a `</think>` block. No output
hit the 16,384-token ceiling.

The reasoning-conditioned scores were almost discrete. Mean confidence was
0.9987 for visible reasoning and 0.9980 for native thinking, compared with
0.8505 for direct logits. The generated decisions agreed with the rescored
argmax on every reasoning row. Thus the extra computation did not preserve or
improve uncertainty in this protocol; it mostly converted a graded direct
margin into a highly confident commitment.

## Interpretation

The primary hypothesis did not pass this pilot. Native thinking was better than
asking the non-thinking model to deliberate visibly, especially on the unified
source, but immediate logits remained stronger in both source-level AUROC and
Brier score. The native-thinking versus direct differences are not resolved by
this 100-row bootstrap, while the visible-reasoning regression is large enough
that its interval excludes zero.

Do not generalize this result to the missing final in-distribution test sets or
the six OOD families. Only one stochastic generation seed was evaluated, and
the source canary is small. The next useful reasoning experiment should test
multiple independent traces per row and aggregate their terminal probabilities;
that would measure uncertainty across reasoning trajectories instead of asking
one sampled trace to remain uncertain after it has argued itself toward a
decision. Run that only after acquiring the exact STRIDE test and
Gloom-Exfiltration artifacts or freezing a defensible replacement split.
