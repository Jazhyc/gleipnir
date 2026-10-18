# SHADE-Arena length audit

Status: exploratory post-hoc diagnostic, completed 2026-10-18.

## Finding

The strict-OOD SHADE-Arena source is a length-truncated subset, and monitor
quality falls with length inside it. Sinha et al.'s 28,000-token input filter
(Qwen3-8B tokenizer) removed 650 of 1,500 SHADE-Arena rows. Other OOD sources
lost at most 4%:

| OOD source | Raw | Dropped for length | Kept |
| --- | ---: | ---: | ---: |
| SHADE-Arena | 1,500 | 650 (43%) | 850 |
| IaC | 1,500 | 57 (4%) | 1,443 |
| CUA-SHADE-Arena | 968 | 13 (1%) | 955 |
| MLE-Sabotage | 151 | 3 (2%) | 148 |
| Rogue Eval | 1,500 | 1 | 1,499 |
| AgentDojo | 1,500 | 0 | 1,500 |

Source: upstream `filtered_eval/filter_log.json` at
`adityaasinha28/deliberative-monitor-pipeline@2f10f4e`. The same cap applied
to the training sets and follows from the paper's 32,768-token SFT context; it
was not a SHADE-specific choice. The dropped rows are not published at that
revision, so their lengths and labels are unknown.

Treat strict-OOD SHADE-Arena scores as optimistic for the full benchmark,
especially for the trained 27B. They say nothing about monitoring above about
28K tokens.

## Length dependence in the retained rows

SHADE-Arena pAUROC@20 by length quartile (Qwen tokens, binary student prompt),
with bootstrap 95% intervals. The stratified view bins by length rank within
main task x attack generator, so all bins share the same task/generator mix.

| Arm | All | Raw Q1 (3.3K-16.5K) | Raw Q4 (23.8K-29.3K) | Stratified Q1-Q4 |
| --- | ---: | ---: | ---: | --- |
| Qwen3.8-27B trained s0 | 0.588 | 0.766 [0.68, 0.85] | 0.494 [0.41, 0.58] | 0.692, 0.721, 0.541, 0.492 |
| Kimi K3 | 0.574 | 0.663 [0.56, 0.78] | 0.499 [0.41, 0.58] | 0.577, 0.702, 0.552, 0.495 |
| Gleipnir 4B s0 | 0.238 | 0.263 | 0.176 | 0.250, 0.351, 0.207, 0.178 |

- **Trained 27B:** the decline survives stratification (stratified Q1 to Q4
  `-0.200`, intervals `[0.59, 0.79]` versus `[0.41, 0.58]`). Within tasks it
  comes from the Cabo expense task: short half `0.997`, long half `0.862
  [0.71, 0.96]`. The email/calendar task is near-perfect in both halves, and the
  Boston conference task is flat (`0.651`/`0.662`).
- **Kimi K3:** about half the raw decline is task/generator mix (stratified
  `-0.082`, non-monotone, overlapping intervals). It also degrades on Cabo
  (`0.967` to `0.874`) but improves with length on Boston (`0.400` to `0.561`).
- **Gleipnir 4B:** weak at every length, and worse on the long half of all
  three large tasks.

Length alone barely predicts the label (SHADE-Arena label AUROC `0.544`).
Long-prompt degradation is mostly specific to SHADE-Arena. Raw Q1-to-Q4
changes elsewhere are at most `-0.057` for the 27B and Kimi on CUA-SHADE, IaC
and MLE-Sabotage. The exception is the 4B on IaC, which falls from `0.904` to
`0.504` in its longest quartile.

## Task difficulty and pooling

SHADE-Arena's low pooled score is mostly one task and cross-task calibration,
not uniform difficulty. Per main task, the trained 27B scores `1.000` on
email/calendar (231 rows), `0.930` on Cabo (267) and `0.613` on the Boston
conference task (333). Kimi K3 scores `1.000`, `0.921` and `0.391`. The 27B's pooled pAUROC@20
(`0.588`) is below all three, so negatives scored high on one task outrank
positives on another; Kimi's pooled `0.574` is dragged mainly by Boston. Generators
differ in length: Gemini 2.5 Pro trajectories are markedly shorter (Cabo median
9,994 tokens versus about 22,800 for Claude Sonnet 4 and GPT-5).

## Limitations

Observational only: length is not manipulated. Within a task, longer
trajectories may differ in side task, attack subtlety or agent behaviour. The
side task is not recorded for negatives, so strata are main task x generator
only. Kimi K3 saw the teacher prompt, but bins use the student-prompt Qwen
count, which orders rows almost identically. Seed 0 only for the trained arms.
The [4B padding counterfactual](monitoring_length_shortcut_audit.md) found
small position sensitivity on ID data; this audit does not test whether the
27B's long-Cabo drop is a context-length effect or a content effect.

## Reproduction

`python -m experiments.shade_length_audit.analyze` writes
`results/shade_length_audit/summary.json` (SHA-256
`6bd1ad3009a0044ac86067091affba40e4cf226f3c02c4c6b32f97b5f436b743`) from:

- `results/qwen38-27b-s0-ood01/evaluation/ood.jsonl`;
- `results/tool_trajectory_monitoring/kimi_k3_ood_{canary,full_remaining}_fireworks.jsonl`;
- `results/teacher-seed-sdpa-s0-rescore01/evaluation/ood.jsonl`;
- `data/teacher-disagreement-strict-ood01/workload.jsonl` and the upstream
  `filtered_eval/ctrl_shade.parquet`.
