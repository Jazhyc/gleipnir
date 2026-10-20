# Gleipnir 4B → 27B cascade

Gleipnir 4B v0.3 scores every row. A routing rule escalates some rows to
Gleipnir 27B, which runs under its full teacher prompt, and a merge rule
combines the two scores. Both models use optimized serving.

- **Cost.** Each row pays the 4B prompt at its self-hosted break-even rate.
  Escalated rows also pay the 27B teacher prompt at its rate. The rates are
  one B200 at $7.99/h, 100% utilized, at the benchmark throughputs: 226K and
  53.9K prompt tokens/s.
- **Tuning.** Everything is tuned on the 3,012 ID rows only
  (`cascade.py` → `results/cascade/id_sweep.json`). OOD is reserved for the
  strategies chosen from this sweep. ID prompts are longer than OOD prompts
  (11.2K/12.0K mean tokens against 8.7K/9.5K), so ID costs run about 30%
  higher than benchmark costs.
- **Rules swept.**
  - Routing: smallest 4B logit margin first (`uncertain`); highest 4B score
    first (`top`); two-sided 4B score-quantile bands.
  - Merge: the 27B score replaces the 4B score (`replace`); the mean of the
    two logits (`logit_average`); the escalated set keeps its 4B values,
    reordered by the 27B (`reorder`).
  - Escalation fractions 2–100%.
- **Selection optimism.** Repeated (5×) stratified two-fold splits within ID
  choose the cheapest fraction within a margin of the 27B on one half and
  score the other half.

## ID results

References: 4B alone 0.9117 ID macro pAUROC@20 ($0.110/1K on ID prompts);
27B alone 0.9444 ($0.495/1K).

ID macro pAUROC@20 by escalated fraction:

| Strategy | 5% | 15% | 30% | 50% | 60% | 100% |
|---|---:|---:|---:|---:|---:|---:|
| uncertain / logit_average | 0.9174 | 0.9287 | **0.9430** | **0.9495** | 0.9503 | 0.9502 |
| uncertain / reorder | 0.9147 | 0.9266 | 0.9392 | 0.9470 | 0.9472 | 0.9444 |
| uncertain / replace | 0.9095 | 0.9099 | 0.9208 | 0.9377 | 0.9423 | 0.9444 |
| top / logit_average | 0.9117 | 0.9117 | 0.9116 | 0.9251 | 0.9394 | 0.9502 |
| Cascade cost per 1K (ID prompts) | $0.139 | $0.196 | $0.284 | $0.393 | $0.448 | $0.605 |

- **Routing on uncertainty wins.** Escalating the rows closest to the 4B's
  decision boundary recovers the 27B's ranking far faster than escalating
  the top-scored rows. Top-first routing changes nothing until about 40%,
  because the 4B's confident extremes are already well ranked.
- **Logit averaging beats replacing.** Replacing interleaves the 27B's
  mid-range scores with the 4B's confident extremes. At small fractions it
  is worse than the 4B alone (0.9065 at 10%). Averaging the two logits
  avoids this and is the best merge at every fraction.
- **Averaging exceeds the 27B alone.** From about 40% escalation, the
  averaged cascade beats the 27B (0.9495 at 50%, 0.9503 at 60%). Averaging
  both models on every row gives 0.9502. On ID the two models are partly
  complementary.
- **Two-sided bands** (for example 4B score quantiles 0.4–0.7) are slightly
  cheaper at matched fractions, but score below the margin-based rule.
- **Cheapest within 0.5 points of the 27B:** `uncertain/logit_average` at
  30% escalation, 0.9430 at $0.284/1K, 43% below the 27B's cost. In
  cross-validation the chosen fraction averages 0.29, with held-out
  pAUROC@20 0.9407 (mean gap to the 27B 0.34 points, worst split 1.8).

## Frozen rule for OOD

Escalation must be frozen as a 4B **margin threshold**, not a fraction,
because the OOD margin distribution differs from ID. The thresholds behind
the ID fractions are:

| ID fraction | Escalate when 4B logit margin is at most | 4B score band |
|---:|---:|---:|
| 15% | 1.375 | 0.202–0.798 |
| 30% | 2.000 | 0.119–0.881 |
| 50% | 2.625 | 0.068–0.932 |

## OOD results (six pre-registered conditions, all reported)

`ood.py` applies the frozen margin thresholds and two merge rules to the
6,395-row OOD suite (optimized serving for both models). Costs are on the
benchmark prompts, at the same break-even rates (4B alone $0.0850/1K, 27B
alone $0.3896/1K).

| Condition (ID fraction, margin ≤) | Merge | Escalated on OOD | Cost / 1K | Mean-OOD pAUROC@20 | AUROC | Brier |
|---|---|---:|---:|---:|---:|---:|
| 4B alone | — | 0% | $0.0850 | 0.7703 | 0.9138 | 0.1579 |
| 15%, 1.375 | logit average | 19.0% | $0.1554 | **0.8099** | 0.9221 | 0.1269 |
| 15%, 1.375 | reorder | 19.0% | $0.1554 | 0.7986 | 0.9196 | 0.1435 |
| 30%, 2.000 | logit average | 34.0% | $0.2105 | 0.8362 | 0.9285 | 0.1052 |
| 30%, 2.000 | reorder | 34.0% | $0.2105 | **0.8393** | 0.9291 | 0.1330 |
| 50%, 2.625 | logit average | 49.2% | $0.2833 | 0.8604 | 0.9427 | 0.0852 |
| 50%, 2.625 | reorder | 49.2% | $0.2833 | **0.8893** | 0.9489 | 0.1250 |
| 27B alone | — | 100% | $0.3896 | 0.9508 | 0.9841 | 0.0458 |

| Per-source pAUROC@20 | AgentDojo | CUA-SHADE | IaC | MLE | Rogue | SHADE-Arena |
|---|---:|---:|---:|---:|---:|---:|
| 4B alone | 0.8401 | 0.7493 | 0.8214 | 1.0 | 0.9586 | 0.2524 |
| 30%, logit average | 0.9720 | 0.7622 | 0.8991 | 1.0 | 0.9687 | 0.4150 |
| 30%, reorder | 0.9766 | 0.7554 | 0.8842 | 1.0 | 0.9677 | 0.4522 |
| 50%, logit average | 0.9810 | 0.7770 | 0.9435 | 1.0 | 0.9807 | 0.4801 |
| 50%, reorder | 0.9900 | 0.7690 | 0.9423 | 1.0 | 0.9799 | 0.6547 |
| 27B alone | 0.9917 | 0.8903 | 0.9931 | 1.0 | 0.9980 | 0.8316 |

- **The ID result did not transfer.** The 30% condition that matched the 27B
  on ID reaches 0.84 on OOD, 11 points below the 27B. At 49% escalation the
  best cascade reaches 0.889.
- **The 4B's uncertainty misses its OOD errors.** Escalation rates track
  the 4B's confidence, not its accuracy: at the 30% threshold only 20.6% of
  CUA-SHADE and 30.5% of SHADE-Arena escalate. Those sources keep most of
  their 4B deficit (CUA-SHADE 0.76 against 0.89, SHADE-Arena 0.42–0.45
  against 0.83). Rows the 4B is confidently wrong about never reach the 27B.
- **Merge rule.** As expected for a much stronger second model, trusting the
  27B's ordering (`reorder`) wins at 30% and 50%. Averaging wins only at 15%,
  where little is escalated. Averaging gives much better Brier at every
  threshold, because it doesn't reuse 4B score values.
- **Frontier.**
  - pplx-decider v1.2 ($0.176, 0.8956) dominates the 30% and 50% conditions.
  - The 15% logit-average condition ($0.155, 0.8099) is cheaper than the
    Decider and better than the 4B, so on point estimates it would sit on
    the frontier between them. It was not the ID-chosen condition (that was
    30%), and choosing it after seeing these results is selection on OOD.
