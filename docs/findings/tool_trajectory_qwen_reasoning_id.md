# Qwen3.5 reasoning on held-out in-distribution trajectories

Date: 2026-08-30. Status: completed direct-logit and visible-reasoning arms.
The native-thinking arm was cancelled before evaluation because of its projected
inference cost.

## Design

The evaluation used all 3,012 frozen held-out rows: 946 STRIDE-test rows and
2,066 Gloom-Exfiltration rows. Both arms used
`Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` in BF16 with vLLM
0.24.0 on one H100 80GB. The direct arm measured the normalized literal `0|1`
logits immediately. The visible-reasoning arm disabled native thinking, sampled
a deliberation under the frozen 16,384-token cap, and measured the same logits
at its terminal decision boundary.

pAUROC@20 is the paper's raw normalized partial area:

```text
(1 / 0.2) * integral from FPR=0 to 0.2 of TPR(FPR) dFPR
```

The implementation clips and linearly interpolates the ROC at FPR 0.2 before
trapezoidal integration. It does not use scikit-learn's standardized partial-AUC
correction.

## Results

| Population | Direct pAUROC@20 | Visible pAUROC@20 | Visible minus direct |
| --- | ---: | ---: | ---: |
| Gloom-Exfiltration | **0.5685** | 0.4790 | -0.0895 |
| STRIDE test | **0.7584** | 0.6381 | -0.1203 |
| Unweighted source macro | **0.6634** | 0.5586 | -0.1049 |
| Pooled | **0.5896** | 0.5314 | -0.0582 |

The corresponding source-macro AUROC was 0.8986 direct versus 0.8216 visible,
and macro Brier was 0.1571 direct versus 0.2727 visible. Visible reasoning
therefore harmed both full-range ranking and the paper-comparable low-FPR
ranking. The low-FPR macro regression was larger than the full AUROC regression
(-0.1049 pAUROC@20 versus -0.0770 AUROC).

Visible reasoning generated 1,060,108 tokens. Five rows lacked a parseable
generated terminal decision, but all rows retained a valid measured `0|1` logit
boundary; no generation was truncated. The non-thinking template produced a
literal `</think>` substring in 46 rows, which does not indicate that native
thinking mode was enabled.

## Operational note

vLLM stalled once after checkpointing 1,776 visible-reasoning rows. The exact
same pinned engine, model, config, and prediction cache resumed successfully and
completed the remaining rows. This does not change row membership or scoring,
but future Qwen3.5 hybrid-model campaigns should canary newer vLLM releases and
prefix-cache-disabled serving before changing the frozen backend.

## Updated belief

For this unadapted Qwen3.5-9B monitor, a single sampled visible deliberation is
not an uncertainty-improving transformation. Immediate decision-boundary logits
are the stronger baseline on both held-out in-distribution sources, including in
the low-FPR region used by the paper. Future reasoning experiments should
estimate uncertainty across multiple independent traces rather than relying on
the terminal confidence of one deliberation.
