# Qwen3.5-9B visible-reasoning ordinal OOD benchmark

Status: completed 2026-08-31.

## Result

On the frozen six-source, 6,395-row strict OOD suite, unadapted
`Qwen/Qwen3.5-9B` reached **0.6656 Mean-OOD pAUROC@20** and **0.8610
Mean-OOD AUROC** when it visibly analyzed each trajectory and then generated a
0--10 severity score. The matched immediate ordinal arm reached `0.4858` and
`0.7592`; visible reasoning therefore improved pAUROC@20 by `0.1798` and AUROC
by `0.1017`.

The continuous immediate binary-logit baseline remained stronger at `0.6976`
pAUROC@20 and `0.8850` AUROC. Visible reasoning left deficits of `0.0320` and
`0.0240`, respectively. It recovered 84.9% of the immediate ordinal arm's
pAUROC@20 gap to binary logits, but did not close it.

The primary paper-comparison endpoint is raw normalized pAUROC over the
`FPR <= 0.20` region, computed independently for each source and averaged
without source-size weighting. Standard AUROC is secondary.

| Source | Reasoned pAUROC@20 | Immediate pAUROC@20 | Binary pAUROC@20 | Reasoned AUROC | Immediate AUROC | Binary AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AgentDojo | 0.5189 | 0.5600 | 0.5657 | 0.8946 | 0.8591 | 0.8792 |
| CUA-SHADE-Arena | 0.6461 | 0.6192 | 0.7814 | 0.8256 | 0.7898 | 0.8981 |
| IaC | 0.7690 | 0.4730 | 0.7348 | 0.8904 | 0.7813 | 0.9257 |
| MLE-Sabotage | 0.9265 | 0.5684 | 0.9602 | 0.9592 | 0.7602 | 0.9824 |
| Rogue Eval | 0.8218 | 0.5501 | 0.7225 | 0.9270 | 0.7537 | 0.9075 |
| SHADE-Arena | 0.3116 | 0.1442 | 0.4212 | 0.6691 | 0.6114 | 0.7171 |
| **Mean-OOD** | **0.6656** | **0.4858** | **0.6976** | **0.8610** | **0.7592** | **0.8850** |

Reasoning exceeds binary pAUROC@20 on IaC and Rogue Eval, but loses on the
other four sources. In particular, the remaining low-FPR deficit is not a
uniform small shift: CUA-SHADE-Arena and SHADE-Arena remain `0.1353` and
`0.1096` below binary, while Rogue Eval improves by `0.0993`.

At threshold 5, the reasoned arm's source-macro balanced accuracy is `0.8328`,
recall is `0.8198`, false-positive rate is `0.1543`, and Brier score is
`0.1570`. The binary baseline's corresponding values are `0.7220`, `0.5394`,
`0.0954`, and `0.1906`. The reasoned scores improve this threshold and Brier
diagnostic while still ranking worse in the primary low-FPR region.

## Score use and paired behavior

The reasoned arm uses all 11 values, unlike the immediate arm's seven values:

| Score | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rows | 2,894 | 27 | 33 | 255 | 44 | 129 | 347 | 1,189 | 1,059 | 96 | 322 |

There are 11 unique scores and 6,384 tied rows, the same resolution as the
nearest-tenth binary control. The reasoned and immediate scores agree exactly
on 50.1% of rows, have mean absolute normalized difference `0.2194`, and
Pearson correlation `0.5945`. This supports the interpretation that the gain
comes from a materially different rubric application, not merely emitting
more of the available bins.

The saved binary probabilities rounded to tenths reach `0.6835` pAUROC@20.
Thus visible reasoning narrows the generated-interface gap substantially, but
still does not match even this 11-bin control (`-0.0179`), much less the
continuous binary probabilities (`-0.0320`).

## Frozen serving contract and boundary audit

The evaluation reuses the exact model revision, OOD rows, labels, full
monitoring role, native non-thinking chat boundary, and temperature-zero
decoding of both baselines. The monitor instruction directly states the five
severity bands; it does not mention the paper or its authors.

Qwen native thinking remains disabled. The first vLLM continuation greedily
generates a visible evidence analysis through the terminal `Score:` prefix.
The second continuation of the same assistant response is constrained to one
of the exact strings 0 through 10. This is logically one rationale and score;
the split prevents formatting failures from becoming missing predictions.

The 4,096-token rationale reserve produced 6,374 normal boundaries, 20
length-cap recoveries, and one completed response missing the exact boundary.
For those 21 rows (0.328%), the evaluator retained every generated analysis
token, appended only `Score:`, recorded the recovery, and requested the
constrained score without regenerating or dropping the row. No response
contained a native `</think>` close. The rationale mean was 440.64 tokens,
with a range of 87--4,096; logical visible completions, including score and
termination, averaged 442.65 tokens.

The instruction SHA-256 is
`f79026c915e85e00c1f42790e9ab415f2f8221b166c42ce36a8594a9d72241fa` and
the frozen config SHA-256 is
`eb840794d5dd1cc4070298c4c6a6aa5fa957431e2ffb5c35460930657b2aabc3`.
Rendered initial inputs totaled 61,575,609 tokens, averaging 9,628.7113 per
row, with a 1,195 minimum and 30,544 maximum. The largest score context was
34,035 tokens under the frozen 36,864-token engine limit.

## Cost and interpretation

At the 2026-08-30 OpenRouter snapshot of `$0.10/M` input and `$0.15/M` output,
the uncached paper-comparable projection is `$0.962871/1K` for input plus
`$0.066397/1K` for output, or **`$1.029268/1K evaluations`**. It costs 8.7%
more than binary logits and 6.9% more than immediate ordinal generation while
remaining below binary pAUROC@20, so it is strictly dominated and is not an
observed frontier point.

This result supports a narrow conclusion: one greedy visible deliberation in
Qwen3.5-9B's non-thinking mode recovers most, but not all, of the generated
ordinal interface's low-FPR ranking loss. It does not show that reasoning is
useless, nor does it test native thinking, sampled self-consistency, trained
rationales, or the paper's fine-tuned monitors.

## Run audit

The persistent vLLM 0.24.0 engine used BF16 upstream weights on one NVIDIA RTX
PRO 6000 Blackwell Server Edition with PyTorch 2.11.0+cu130. The final balanced
canary was Slurm job `30979591`. Definitive job `30979752` completed all 6,395
rows and result assembly in 2:08:44 with exit code zero.

Earlier preflights were invalidated before analysis. Jobs `30978661` and
`30978763` exposed a repeated 2,048-token cap; job `30978885` showed the same
example could exhaust 4,096 tokens; and job `30979145` exposed an accounting
assertion caused by tokenizer merges across the continuation boundary. These
runs motivated the frozen bounded-boundary policy and retokenized logical
completion accounting. They were not combined with the definitive artifact.

- Predictions SHA-256: `f0b8b5ff17552fa7331a886f54d29613fdc67e413ab5609ade121b524e03d7b3`
- Canary SHA-256: `e12adaa19a39ca6d1be91b86e71142369fc556c2ce5c26ebbd62fdd7a35d552c`
- Result SHA-256: `2b32d5294d4cf1443707001604805db1157cc20c8b4858caa523474a193f3461`

The ignored artifacts are under
`results/tool_trajectory_monitoring/qwen35_9b_teacher_ordinal_reasoning_ood_v3_final/`
and can be reconstructed with the frozen config and Slurm entrypoint.

## Sources

- [Sinha et al., score rubric and monitor evaluation](https://arxiv.org/html/2605.29601v1)
- [OpenRouter Qwen3.5-9B pricing](https://openrouter.ai/qwen/qwen3.5-9b)
- [`tool_trajectory_qwen_ordinal_ood_benchmark.md`](tool_trajectory_qwen_ordinal_ood_benchmark.md)
- [`tool_trajectory_ood_frontier.md`](../research/tool_trajectory_ood_frontier.md)
