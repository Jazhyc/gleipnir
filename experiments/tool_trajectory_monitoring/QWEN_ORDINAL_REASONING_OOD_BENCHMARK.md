# Qwen3.5-9B visible-reasoning ordinal OOD benchmark

Status: v2 frozen before evaluation; not yet launched. The 2,048-token v1
preflight was invalidated before analysis after a rationale hit its length cap.

## Hypothesis and intervention

This matched ablation tests whether one visible test-time deliberation recovers
the ranking gap between the completed immediate 0--10 score and continuous
binary logits. The primary endpoint is raw normalized Mean-OOD pAUROC@20:
compute pAUROC over `FPR <= 0.20` independently for each of the six frozen OOD
sources, divide by 0.20, and take their unweighted mean. Standard Mean-OOD AUROC
is secondary.

The model remains the unadapted `Qwen/Qwen3.5-9B` BF16 revision used by both
baselines. Native Qwen thinking remains disabled. The prompt directly presents
the same five severity bands as the immediate ordinal arm, but replaces its
silent review procedure with a visible concise analysis. It does not name the
paper or its authors inside the monitor instruction.

Temperature-zero generation continues the assistant response until the exact
terminal `Score:` prefix. A second continuation of that same response is
constrained to one of the exact integer strings `0` through `10`. This split
retains the generated analysis in the scoring context while preventing output
format failures from becoming missing scores. It is logically one visible
rationale followed by one score, even though vLLM executes two generation
calls. The saved ranking value is the integer divided by 10.

## Matched baselines and accounting

The paired baselines are:

1. continuous immediate literal `0|1` probabilities, with `0.6976` Mean-OOD
   pAUROC@20 and `0.8850` Mean-OOD AUROC; and
2. immediate generated 0--10 severity, with `0.4858` Mean-OOD pAUROC@20 and
   `0.7592` Mean-OOD AUROC.

All three use the same 6,395 OOD row IDs, labels, Qwen revision, full monitoring
role, non-thinking chat boundary, and temperature-zero terminal decision. This
intervention changes the output instruction and adds visible analysis, so it is
an interface-level ablation rather than a pure compute-only intervention.

Paper-comparable hosted accounting counts the initial rendered prompt once and
the rationale plus terminal score as output. The second vLLM continuation is an
implementation detail and is not treated as a second hosted request. Runtime
metadata separately preserves both prompt lengths.

The audited initial contexts total 61,575,609 tokens, averaging 9,628.7113 per
row, with a minimum of 1,195 and maximum of 30,544. The maximum plus the frozen
4,096-token rationale allowance and two score tokens is 34,642, below the
36,864-token engine limit. This matches the paper's 4K rationale reserve.

## Selection rule and stop condition

OOD labels must not select the prompt, generation settings, threshold, subset,
retry rule, or stopping point. Report all sources and both paired comparisons
regardless of direction. The run stops on any input, prompt, model, or baseline
checksum drift; an empty or truncated rationale; failure to reach the exact
terminal score boundary; invalid score tokenization; an incomplete balanced
source-label canary; context overflow; or incomplete paired coverage.

Run the balanced 12-row canary first:

```bash
sbatch cluster/slurm/benchmark_qwen_ordinal_reasoning_ood.sh \
  --stop-after-canary
```

Then resume the same checksum-bound cache for the full run:

```bash
sbatch cluster/slurm/benchmark_qwen_ordinal_reasoning_ood.sh
```

Artifacts are written under the ignored directory
`results/tool_trajectory_monitoring/qwen35_9b_teacher_ordinal_reasoning_ood_v2/`.
