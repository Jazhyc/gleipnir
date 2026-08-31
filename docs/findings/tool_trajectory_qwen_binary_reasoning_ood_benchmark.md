# Qwen3.5-9B visible-reasoning binary-logit OOD benchmark

Status: completed 2026-08-31.

## Result

On the frozen six-source, 6,395-row strict OOD suite, unadapted
`Qwen/Qwen3.5-9B` reached **0.6923 Mean-OOD pAUROC@20** and **0.8607
Mean-OOD AUROC** after one visible analysis followed by normalized literal
`0`/`1` logits. The matched immediate binary-logit baseline reached `0.6976`
and `0.8850`, respectively. Visible reasoning therefore changed the primary
metric by `-0.0054` and AUROC by `-0.0243`; it did not improve the immediate
logits.

The primary paper-comparison endpoint is raw normalized pAUROC over the
`FPR <= 0.20` region, computed independently for each source and averaged
without source-size weighting. Standard AUROC is secondary.

| Source | Reasoned pAUROC@20 | Immediate pAUROC@20 | Delta | Reasoned AUROC | Immediate AUROC | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AgentDojo | 0.8637 | 0.5657 | +0.2980 | 0.9617 | 0.8792 | +0.0824 |
| CUA-SHADE-Arena | 0.6590 | 0.7814 | -0.1223 | 0.8430 | 0.8981 | -0.0551 |
| IaC | 0.7103 | 0.7348 | -0.0245 | 0.8535 | 0.9257 | -0.0722 |
| MLE-Sabotage | 0.8786 | 0.9602 | -0.0816 | 0.9337 | 0.9824 | -0.0488 |
| Rogue Eval | 0.7191 | 0.7225 | -0.0034 | 0.8633 | 0.9075 | -0.0442 |
| SHADE-Arena | 0.3229 | 0.4212 | -0.0983 | 0.7093 | 0.7171 | -0.0077 |
| **Mean-OOD** | **0.6923** | **0.6976** | **-0.0054** | **0.8607** | **0.8850** | **-0.0243** |

The near-flat macro pAUROC hides a large redistribution. Reasoning improves
AgentDojo by `0.2980`, but declines on the other five sources. The intervention
therefore does not provide a source-robust recovery even though it is nearly
matched on the unweighted mean.

At threshold 0.5, the reasoned arm's source-macro balanced accuracy is
`0.8119`, recall is `0.7376`, false-positive rate is `0.1139`, and Brier score
is `0.1910`. The direct baseline's corresponding values are `0.7220`, `0.5394`,
`0.0954`, and `0.1906`. Visible reasoning improves the fixed-threshold
balanced-accuracy diagnostic by raising recall, but it also raises false
positives and does not improve calibration or the primary low-FPR ranking
metric.

## Paired score behavior

The reasoned logits produce 6,314 unique probabilities with 81 tied rows,
compared with 1,587 unique probabilities and 4,808 tied rows for immediate
logits. More numerical resolution does not translate into better ranking.

Across paired rows, terminal hard decisions agree on 79.16% of examples. The
mean absolute probability change is `0.2721`, and Pearson score correlation is
`0.6895`. Visible analysis therefore materially changes the model's uncertainty
estimate rather than merely perturbing the same boundary, but the change is not
reliably aligned with OOD labels.

The matched reasoned ordinal arm reached `0.6656` pAUROC@20 and `0.8610` AUROC.
Keeping continuous binary logits after reasoning improves its primary metric by
`0.0267`, while AUROC is essentially unchanged. This cross-interface comparison
is secondary because both the terminal representation and instruction differ.

## Frozen serving contract and boundary audit

The evaluation reuses the immediate baseline's exact model revision, OOD rows,
labels, full binary security specification, native non-thinking chat boundary,
literal decision tokens, and temperature-zero terminal decision. The monitor
instruction states the scoring semantics directly and does not mention the
paper or its authors.

Qwen native thinking remains disabled. The first vLLM continuation greedily
generates a visible evidence analysis through the terminal `Prediction:` prefix.
The second continuation of the same assistant response requests both literal
`0` and `1` logprobs and normalizes them with the same two-class softmax as the
direct baseline. This is logically one rationale and decision; the split makes
the decision boundary exact.

The 4,096-token rationale reserve produced 6,380 normal boundaries, 13
length-cap recoveries, and two completed responses missing the exact terminal
boundary. For those 15 rows (0.235%), the evaluator retained every analysis
token, appended only `Prediction:`, recorded the recovery, and extracted the
same constrained logits without regenerating or dropping the row. No response
contained a native `</think>` close. Rationales averaged 386.06 tokens, with a
range of 98--4,096; logical completions including the boundary and decision
averaged 387.07 tokens.

The instruction SHA-256 is
`ee612b3844446a6a085cdcd34d25fd5f9b33ae8948b2ad6c1965ea0564edd491`
and the frozen config SHA-256 is
`2bd4cbfc890ad16888c6efb9628bc7f66fb4b1513ee7096656a15207bd670292`.
Rendered initial inputs totaled 60,584,384 tokens, averaging 9,473.7113 per
row, with a 1,040 minimum and 30,389 maximum. The largest realized decision
context was 31,279 tokens; the audited worst-case bound of 34,486 remained
below the frozen 36,864-token engine limit.

## Cost and interpretation

At the 2026-08-30 OpenRouter snapshot of `$0.10/M` input and `$0.15/M` output,
the uncached paper-comparable projection is `$0.947371/1K` for input plus
`$0.058061/1K` for output, or **`$1.005432/1K evaluations`**. It costs 6.2%
more than immediate binary logits while ranking slightly worse, so it is
strictly dominated and does not enter the observed frontier.

This result supports a narrow conclusion: one greedy visible deliberation in
Qwen3.5-9B's non-thinking mode does not improve the full-specification binary
logit monitor on the primary six-source Mean-OOD pAUROC@20 endpoint. It does
not establish that reasoning is universally harmful, nor does it test native
thinking, sampled self-consistency, trained rationales, or the paper's
fine-tuned monitors.

## Run audit

The persistent vLLM 0.24.0 engine used BF16 upstream weights on one NVIDIA RTX
PRO 6000 Blackwell Server Edition with PyTorch 2.11.0+cu130. The balanced
12-row canary completed as Slurm job `30986516`. Definitive job `30988212`
resumed those rows and completed all 6,395 predictions plus result assembly in
1:59:35 with exit code zero.

- Predictions SHA-256: `27038ac8534a6491265ef5721c52a900aa36cfd16bdec132a3a906fb9b9a6f66`
- Canary SHA-256: `6ea692a841e357e821b0b20043a88e97c41a4220d42f1f7736e462efdc706fd9`
- Result SHA-256: `fafec69d09df21ef87d03a93ac9be4509182c05deadc4858db28c7000a3beb71`

The ignored artifacts are under
`results/tool_trajectory_monitoring/qwen35_9b_teacher_binary_reasoning_ood/`
and can be reconstructed with the frozen config and Slurm entrypoint.

## Sources

- [Sinha et al., monitor evaluation](https://arxiv.org/html/2605.29601v1)
- [OpenRouter Qwen3.5-9B pricing](https://openrouter.ai/qwen/qwen3.5-9b)
- [`tool_trajectory_qwen_ood_benchmark.md`](tool_trajectory_qwen_ood_benchmark.md)
- [`tool_trajectory_qwen_ordinal_reasoning_ood_benchmark.md`](tool_trajectory_qwen_ordinal_reasoning_ood_benchmark.md)
- [`tool_trajectory_ood_frontier.md`](../research/tool_trajectory_ood_frontier.md)
