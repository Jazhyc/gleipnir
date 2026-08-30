# Qwen3.5-9B immediate ordinal-score OOD benchmark

Status: completed 2026-08-30.

## Result

On the frozen six-source, 6,395-row strict OOD suite, unadapted
`Qwen/Qwen3.5-9B` reached **0.4858 Mean-OOD pAUROC@20** and **0.7592
Mean-OOD AUROC** when it generated an immediate integer severity score from 0
through 10. The matched continuous binary-logit baseline reached `0.6976` and
`0.8850`, respectively. The ordinal interface therefore lost `0.2118`
pAUROC@20 and `0.1258` AUROC.

The primary paper-comparison endpoint is raw normalized pAUROC over the
`FPR <= 0.20` region, computed independently for each source and averaged
without source-size weighting. Standard AUROC is secondary. The generated
score was divided by 10 for ranking; the paper's operational threshold of 5
corresponds to a normalized threshold of 0.5.

| Source | Ordinal pAUROC@20 | Binary pAUROC@20 | Delta | Ordinal AUROC | Binary AUROC | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AgentDojo | 0.5600 | 0.5657 | -0.0058 | 0.8591 | 0.8792 | -0.0201 |
| CUA-SHADE-Arena | 0.6192 | 0.7814 | -0.1622 | 0.7898 | 0.8981 | -0.1083 |
| IaC | 0.4730 | 0.7348 | -0.2618 | 0.7813 | 0.9257 | -0.1444 |
| MLE-Sabotage | 0.5684 | 0.9602 | -0.3918 | 0.7602 | 0.9824 | -0.2222 |
| Rogue Eval | 0.5501 | 0.7225 | -0.1723 | 0.7537 | 0.9075 | -0.1538 |
| SHADE-Arena | 0.1442 | 0.4212 | -0.2770 | 0.6114 | 0.7171 | -0.1057 |
| **Mean-OOD** | **0.4858** | **0.6976** | **-0.2118** | **0.7592** | **0.8850** | **-0.1258** |

At threshold 5, the ordinal run's source-macro balanced accuracy was `0.7385`,
recall `0.6176`, false-positive rate `0.1406`, and Brier score `0.2678`. The
binary baseline's corresponding values were `0.7220`, `0.5394`, `0.0954`, and
`0.1906`. The coarse score raises recall but is substantially worse for ranking
and calibration.

## Quantization control

The generated condition changes both numerical resolution and the semantics of
the requested decision. To isolate resolution, the saved binary probabilities
were also rounded to the nearest tenth without rerunning the model.

| Score interface | Unique scores | Tied rows | Mean-OOD pAUROC@20 | Mean-OOD AUROC |
| --- | ---: | ---: | ---: | ---: |
| Continuous binary probability | 1,587 | 4,808 | 0.6976 | 0.8850 |
| Binary probability rounded to nearest tenth | 11 | 6,384 | 0.6835 | 0.8726 |
| Generated ordinal severity | 7 | 6,388 | 0.4858 | 0.7592 |

Nearest-tenth quantization costs only `0.0141` pAUROC@20 and `0.0124` AUROC.
Thus, simple 11-bin ties explain only a small part of the generated ordinal
interface's loss. Most of the gap comes from how the unadapted model maps the
rubric to a generated score, not from discrete resolution alone.

The generated score counts were `0: 3,729`, `3: 195`, `5: 153`, `7: 713`,
`8: 131`, `9: 50`, and `10: 1,424`; scores 1, 2, 4, and 6 were never used.
The extremes 0 and 10 account for 80.6% of all rows. This suggests that listing
bands whose first anchors are 0, 3, 5, 7, and 9 did not produce a calibrated
11-level scale in the base model.

This is evidence against using an **immediate, unadapted generated-severity
interface** as a drop-in replacement for binary logits. It is not evidence
against a trained ordinal monitor or the paper's rationale-before-score
interface.

## Frozen design and serving contract

The evaluation reused the binary run's exact model revision, OOD rows, labels,
full monitoring specification, non-thinking chat boundary, and temperature-zero
decoding. The intervention directly described the five score bands in the
monitor instruction; it did not mention the paper or its authors. The first
assistant content was constrained by vLLM to one of the exact strings `0`
through `10`, with no rationale or label prefix.

Qwen tokenizes 0 through 9 as one token and 10 as two. vLLM completed one-digit
choices with EOS, so every saved response used exactly two output tokens. The
evaluator accepted no trailing content other than that explicit terminator and
recorded the decision and termination token IDs separately.

The instruction SHA-256 is
`3cea884b4426412c29f3b0326e8b37887cea6e851d3d243e8bd83107f2c7d5f0`.
Rendered inputs totaled 61,550,029 tokens, averaging 9,624.7113 per row, with a
1,191 minimum and 30,540 maximum under the frozen 32,768-token context limit.

## Cost and run audit

At the 2026-08-30 OpenRouter snapshot of `$0.10/M` input and `$0.15/M` output,
the uncached paper-comparable projection is `$0.962471/1K` for input plus
`$0.000300/1K` for output, or **`$0.962771/1K evaluations`**. This is 1.7% more
than the binary run's `$0.946821/1K`, because the ordinal instruction is longer.
The ordinal point is dominated on both cost and pAUROC@20 and is not on the
observed frontier.

The persistent vLLM 0.24.0 engine used BF16 upstream weights on one NVIDIA RTX
PRO 6000 Blackwell Server Edition with PyTorch 2.11.0+cu130. A 12-row canary
completed as Slurm job `30976590`. Full job `30976649` generated and atomically
saved all 6,395 valid predictions in 39:30, then failed only in the initial
result-assembly code. After the paired-analysis fix, checksum resume job
`30978023` found zero pending rows, assembled the result, and completed in 0:58.

- Predictions SHA-256: `40864278e45e099731f80d570f78e582c40604c6da081e0b977b2e691a0a3527`
- Canary SHA-256: `8377da0a93f0f833fcf0c07e460ace159e4d3023786095bdd15784188b3e2b64`
- Result SHA-256: `ecf12642f34cb204b935f31a57a47a8ac38b551fd2a38ba7ec8e271244dc7122`

The ignored artifacts are under
`results/tool_trajectory_monitoring/qwen35_9b_teacher_ordinal_ood/` and can be
reconstructed with the frozen config and Slurm entrypoint.

## Sources

- [Sinha et al., score rubric and monitor evaluation](https://arxiv.org/html/2605.29601v1)
- [OpenRouter Qwen3.5-9B pricing](https://openrouter.ai/qwen/qwen3.5-9b)
- [`tool_trajectory_qwen_ood_benchmark.md`](tool_trajectory_qwen_ood_benchmark.md)
- [`tool_trajectory_ood_frontier.md`](../research/tool_trajectory_ood_frontier.md)
