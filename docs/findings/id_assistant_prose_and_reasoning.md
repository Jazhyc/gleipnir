# ID assistant prose and reasoning audit

Status: structural audit and full token accounting completed 2026-09-07.

The frozen 3,012-row ID monitor inputs contain private reasoning. The prior
"action-only" description is inaccurate for these inputs. This finding concerns
input content, not newly measured classifier performance. OOD was excluded from
the cleanup and counts at the user's direction.

## Reasoning presence

All 3,343 nonempty typed `reasoning` blocks in the original STRIDE
`trajectory_data` occur verbatim in the frozen monitor prompts. They occur in
883/946 STRIDE trajectories (93.34%), flattened into assistant prose without
explicit reasoning delimiters. Separately, 7,849 `think` calls occur in 851/946
STRIDE trajectories (89.96%). Their union is 937/946 STRIDE trajectories
(99.05%), or 937/3,012 ID trajectories (31.11%).

Gloom has no explicit `think` calls in the frozen ID transcript. Its available
`trajectory_data` embeds the flat transcript, rather than separately typed
assistant reasoning blocks, so absence of typed reasoning cannot establish
that all remaining tool arguments are semantically free of reasoning.

## Transformation and validation

The opt-in derived ID copy removes all assistant prose, including prose-only
turns and prose accompanying tool calls. A second variant additionally removes
explicit `think` calls. It retains task context, tool results, and all other
calls and their arguments, including submission text. It does not remove
natural-language content nested in arbitrary non-thinking tool arguments.

All 19,735 STRIDE call blocks were matched exactly to the structured source
names/arguments before filtering. This handles mixed think/action turns and
multiline or imperfectly escaped arguments without mistaking embedded code for
another call. None of the 3,343 typed reasoning blocks remain verbatim after
cleanup. The original frozen prompt file and its hashes stay unchanged.

The entrypoint, configuration and reproducibility details are in
[the experiment README](../../experiments/id_assistant_prose_audit/README.md).
Derived prompts are under `data/id_assistant_prose_audit/`; per-row token counts
and the aggregate manifest are under `results/id_assistant_prose_audit/`.
The reference full-input total is 34,631,573 tokens, exactly matching the prior
Gleipnir 4B held-out ID evaluation. Counts include the compact instruction,
Qwen3.5-4B non-thinking chat wrapper and `Prediction:` decision boundary.

## Token savings

| ID source | Original tokens | Remove assistant prose | Also remove think calls | Combined reduction |
| --- | ---: | ---: | ---: | ---: |
| STRIDE test | 5,300,017 | 4,659,498 | 3,911,434 | 26.20% |
| Gloom-Exfiltration | 29,331,556 | 27,972,333 | 27,972,333 | 4.63% |
| Total | 34,631,573 | 32,631,831 | 31,883,767 | 7.93% |

Assistant-prose removal saves 1,999,742 tokens (5.77%); removing explicit think
calls saves another 748,064. Combined savings are 2,747,806 tokens, or 912.29
per trajectory. Mean full-input length falls from 11,497.87 to 10,585.58.
The prose-removal column already removes the flattened typed reasoning blocks;
it is not a pure visible-prose ablation.

Cleaned prompts SHA-256:
`a811f90013442c5f50726bc35c6834fa11c6c0d31e41021aff352a0d928b4847`.
All 3,012 row IDs, source assignments and labels are preserved. Seven focused
parsing/preservation tests and Ruff pass. No training was performed; the subsequent
frozen-checkpoint evaluation is recorded below.

## Implication

Historical ID performance remains valid for the exact original prompts, but
must not be described as a strict no-reasoning/action-only evaluation. The
cleaned-input classifier performance is recorded below. Any later model
comparison needs fresh inference and separate input/cache identities.

## Completed Gleipnir 4B evaluation

On 2026-09-07, the standard mixed-data rank-128 checkpoint
`soft-n21837-mixed-qwen35-4b-seed0` completed all 3,012 cleaned ID examples on
the two reserved Lambda H100s. Both shards completed on attempt zero (1,536 and
1,476 rows). Master/serving parity passed before inference. The local comparison
validated frozen input/config hashes, exact row coverage, finite scores, and
matched IDs, sources and labels against the historical baseline.

| Source | Original pAUROC@20 | Cleaned pAUROC@20 | Original AUROC | Cleaned AUROC |
| --- | ---: | ---: | ---: | ---: |
| STRIDE (946) | 0.966052 | 0.918835 | 0.992866 | 0.981051 |
| Gloom-Exfiltration (2,066) | 0.772335 | 0.687128 | 0.922492 | 0.897623 |
| Source macro | 0.869194 | 0.802982 | 0.957679 | 0.939337 |

Macro pAUROC@20 falls 0.066212. Macro Brier worsens from 0.074303 to
0.103670. At the unchanged 0.5 threshold, macro recall falls from 0.850790
to 0.783150 and macro FPR rises from 0.039826 to 0.053941. There are 226
prediction flips: 156 positive-to-negative and 70 negative-to-positive.
Unique scores fall from 1,631 to 1,573. Full pooled and per-source diagnostics
are in `results/id_action_only_evaluation/comparison.json`.

This establishes sensitivity to the combined removal of assistant prose and
thinking traces, not the isolated effect of private reasoning. Gloom's larger
drop occurs under prose removal alone (no explicit think calls were found).
The model was not retrained for the changed input distribution. No checkpoint
or threshold was selected using these results, and OOD was not evaluated.

Reproduce with `python -m experiments.id_action_only_evaluation.summarize` after
collecting predictions; see the experiment README for the frozen protocol.
Cleaned predictions SHA-256:
`fe22e5a3a8ffca61d9b3ada08e0589d673e147778e81dedaf0c8379da35606d5`.
Baseline predictions SHA-256:
`f62ec78921b008080702f648efd90da0e3d06dd48e3b425eebca1ec23bf5bbde`.

## Follow-up: only identified CoT removed

The completed CoT-only follow-up preserves visible assistant prose and removes
source-typed reasoning blocks plus named `think` calls. All 3,012 rows passed
coverage and frozen-input checks. STRIDE pAUROC@20 changes from 0.966052 to
0.928943; Gloom, whose inputs are unchanged, changes from 0.772335 to 0.771224.
Macro pAUROC@20 is 0.850083, compared with 0.869194 originally and 0.802982
under combined prose/CoT removal. Macro AUROC is 0.952674 versus 0.957679
originally. The much smaller Gloom change is consistent with inference variation.

This confirms sensitivity to identified reasoning on STRIDE, while showing that
the broader removal's decline cannot be assigned wholly to private CoT. It does
not measure a model retrained on cleaned inputs. See the
[CoT-only experiment](../../experiments/id_cot_only_evaluation/README.md) for
per-source diagnostics and preservation checks, including retention of visible
text copies identical to reasoning blocks.
