# Prompt work of monitoring every action

User request, 2026-11-19: how much prompt work prefix caching saves when a
monitor checks every action of real trajectories, as evidence for the
report's serving section. CPU-only token accounting; no GPU, no paid calls.

## Method

- **Checks.** One check before every action: the prompt prefix through the
  end of each tool-calling assistant turn, then the prompt's closing suffix
  (envelope end, chat template, `Prediction:`). Assistant turns without a
  tool call get no check, as in `b200_prefix_replay`.
- **Policies.** *Isolated*: one check of the whole trajectory, the basis of
  every cost in the report. *Uncached*: every check processes its whole
  prompt, as for a model without prefix caching (pplx-decider's
  bidirectional attention). *Cached*: every check processes only the text
  after the previous check's prefix, plus the suffix.
- **Unit.** Characters, as a stand-in for tokens within a prompt; only ratios
  are reported.
- **Workloads.** The 96 agent trajectories of the serving benchmark
  (`data/b200_inference_benchmark`, `agent_only`), and the 6,395-row OOD
  suite (teacher prompts, `data/tool_trajectory_monitoring/teacher_ood_benchmark`).
  148 OOD rows have no rendered tool call and are skipped.
- **Check.** The same count on the synthetic session that
  `b200_prefix_replay` timed (598 checks to 1,037,565 tokens).

```bash
PYTHONPATH=src:. python -m experiments.serving.prefix_caching_tokens.compute
```

## Result (2026-11-19)

| Workload | Trajectories | Checks (median / mean) | Uncached / cached (aggregate) | Uncached / cached (median trajectory) | Cached / isolated |
|---|---:|---:|---:|---:|---:|
| Benchmark, 96 agent trajectories | 96 | 16 / 19.6 | 11.2 | 8.3 | 1.02 |
| OOD suite | 6,247 | 10 / 18.0 | 19.1 | 6.3 | 1.01 |
| OOD: AgentDojo | 1,500 | 4 / 4.3 | 3.9 | 3.5 | 1.01 |
| OOD: CUA-SHADE-Arena | 955 | 22 / 25.7 | 17.6 | 13.5 | 1.01 |
| OOD: IaC | 1,443 | 18 / 19.2 | 12.6 | 10.5 | 1.01 |
| OOD: Rogue Eval | 1,499 | 2 / 2.4 | 2.6 | 2.0 | 1.00 |
| OOD: SHADE-Arena | 850 | 53 / 59.1 | 38.6 | 35.0 | 1.02 |
| Synthetic session (timed) | 1 | 598 | 287.6 | — | — |

- **With caching, monitoring every action costs about one isolated check in
  prompt work** (1.00–1.08 of it); without caching it costs the uncached
  ratio times as much: 11× on the benchmark and 19× on the OOD suite in
  aggregate, from 2.6× (Rogue Eval) to 39× (SHADE-Arena). The OOD suite drops
  trajectories over 28,000 tokens, so longer real sessions would have larger
  ratios.
- **Token work overstates the time saving.** On the synthetic session the
  token ratio is 288×, but the measured time ratio was 12.9×: cached checks
  still pay a fixed per-request cost and attend over the cached prefix
  (median cached check 0.09 s below 64K tokens, against 0.22 s uncached).
  That measurement used the 4B on an earlier stack with YaRN, so it does not
  price the released stack at OOD lengths.
- **Use.** The ratios bound the prompt work of a non-caching monitor such as
  pplx-decider under per-action monitoring. The cost of the cached Gleipnir
  path needs a timed replay of these checks on the released stack.

Outputs: `results/prefix_caching_tokens/summary.json` (ignored).
