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
parsing/preservation tests and Ruff pass. No inference or training was performed.

## Implication

Historical ID performance remains valid for the exact original prompts, but
must not be described as a strict no-reasoning/action-only evaluation. The
cleaned-input classifier performance has not been measured. Any later model
comparison needs fresh inference and separate input/cache identities.
