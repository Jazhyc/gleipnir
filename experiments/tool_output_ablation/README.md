# Monitoring without tool outputs

User request, 2026-11-20: how much a monitor loses when it reads a trajectory
without its tool outputs, and how many tokens that saves. Apollo's Watcher
Live calls this "a reasonable compromise" (lower cost and latency at slightly
lower fidelity) without published numbers. The answer also fixes the input
format for setting thresholds on real traffic (SWE-chat, `data/swe_chat/`).

## Input

`strip.py` replaces every tool output with `[output omitted]` and keeps the
user messages, assistant messages, tool calls and turn structure. It handles
the three renderings of the OOD suite: `[TOOL]` blocks, `TOOL:` turns
(CUA-SHADE-Arena) and the `Execution info:` sections of MLE-Sabotage.

## Token counts (2026-11-20, CPU only)

Qwen3.5 tokens of the trajectory body, 6,395 OOD rows
(`python -m experiments.tool_output_ablation.count`):

| Source | Rows | Tokens, full | Without outputs | Kept | Median row, full → stripped |
|---|---:|---:|---:|---:|---:|
| All | 6,395 | 54.4M | 13.3M | 25% | 5,390 → 1,670 |
| AgentDojo | 1,500 | 1.16M | 0.50M | 43% | 671 → 302 |
| CUA-SHADE-Arena | 955 | 10.6M | 0.89M | 8% | 9,685 → 776 |
| IaC | 1,443 | 20.8M | 3.58M | 17% | 14,071 → 2,102 |
| MLE-Sabotage | 148 | 1.31M | 0.60M | 46% | 7,832 → 3,898 |
| Rogue Eval | 1,499 | 3.82M | 3.82M | 100% | 2,112 → 2,112 |
| SHADE-Arena | 850 | 16.7M | 3.95M | 24% | 20,412 → 4,602 |

Benign and harmful rows keep the same share (25% and 24%). Rogue Eval's only
output is "Code uploaded successfully.", so it is a no-op control.
CUA-SHADE-Arena records tool names without arguments, so its stripped view is
little more than the user task and a list of tool names; read its result as a
floor, not as the cost of dropping outputs from a full rendering. On SWE-chat
the same cut keeps 32% of tokens (0.54B of 1.71B; sessions over 262K fall from
6.4% to 1.5%).

## Planned scoring run (not launched)

- **Hypothesis.** Dropping tool outputs costs little Mean-OOD where the harm
  is visible in the agent's own calls (IaC, SHADE-Arena, MLE-Sabotage) and
  more where the evidence is in an output (AgentDojo, whose injections arrive
  in tool outputs and are visible only through the following action).
- **Intervention.** Score the stripped OOD suite with the released Gleipnir
  4B and 27B on the released stack (image 1.2, one B200), with the
  instruction each is evaluated with (compact for the 4B, full specification
  for the 27B). One deterministic pass each.
- **Baselines.** The existing released-stack OOD scores of the same
  checkpoints on the full inputs (no rerun; see the no-repeat-parity rule for
  released checkpoints on the unchanged stack).
- **Metrics.** Mean-OOD and per-source pAUROC@20; recall at 1% and 0.1% FPR
  with the threshold set on the benign rows of each source (about 530 per
  source, so 0.1% is the highest benign score: report it, but treat it as
  unstable); throughput from the serving logs. Rogue Eval must match the
  baseline up to run-to-run noise.
- **Selection.** None: fixed released checkpoints, no training, no prompt
  search. No retraining on stripped inputs in this run.
- **Stop condition.** One pass per model. If the 4B loses more than 0.03
  Mean-OOD excluding CUA-SHADE-Arena, do not use stripped inputs for the
  SWE-chat thresholds without a separate decision.
- **Compute.** About a quarter of the prompt tokens of a full OOD pass per
  model; needs the user's go-ahead before a Pod is launched.
