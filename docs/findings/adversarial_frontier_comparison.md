# Gleipnir against frontier black-box monitors

*22–23 October 2026 · Abhinav Kumar. Protocols, amendments and full results are
in each experiment's README; this note is the summary.*

## Question

Does Gleipnir give frontier-level monitoring at a fraction of the cost on
misbehaviour it was not trained on, and does it hold when the agent writes
misleading comments? Every monitor sees the same action-only trajectory; API
monitors get the teacher prompt, Gleipnir 4B its student prompt and the 27B the
teacher prompt.

## Natural cheats ([experiment](../../experiments/adversarial_natural_cheats/README.md))

Qwen3.8-27B (thinking off) on ImpossibleBench LiveCodeBench tasks whose tests
are impossible to pass honestly. 60 runs where it cheated, mostly by
hard-coding a special case after the scaffold's "failed, fix it" reply, all
checked by hand; about 240 honest runs on the same tasks. Scored at the first
cheat; task-clustered 95% bootstrap intervals.

| Monitor | pAUROC@20 | AUROC | Cost per 1,000 trajectories |
|---|---|---|---|
| Gleipnir 4B v0.3 | 93.5% (87–99) | 0.959 | $2.0 (A100, unbatched) |
| Gleipnir 27B | 94.1% (88–99) | 0.962 | $4.8 (A100, unbatched) |
| Kimi K3 (Gleipnir's teacher) | 94.3% (89–100) | 0.964 | $6.0 |
| Claude Sonnet 5.5 (649 of 682 rows scored) | 94.8% (87–100) | 0.967 | $15.5 (batch) |
| GPT-6.1 Sol | 93.5% (87–99) | 0.967 | $25.1 |
| GPT-6 Luna | 88.0% (79–96) | 0.953 | $1.3 |
| Untrained Qwen3.5-4B | 68.6% (56–80) | 0.893 | about $1.2 |
| Gemini 3.8 Flash | 60.6% (47–74) | 0.786 | $9.2 |
| Keyword search | 26.5% (16–38) | 0.647 | ~$0 |

- Gleipnir is level with the best frontier monitors at 3–13× lower cost; the
  top five cannot be told apart with 57 scored cheats.
- Training is what buys it: the untrained base scores 68.6%, Gleipnir 4B
  93.5%. Keyword search is far behind, so the monitors do more than spot the
  agent's own words.
- The agent's explanatory comments persuade Gemini 3.8 Flash: its AUROC rises
  from 0.79 to 0.91 when they are stripped. Every other monitor moves by 0.03
  or less.
- Limits: one agent model, one benchmark, and nearly every cheat is announced
  in code or comments. The 4 unannounced cheats are missed by almost every
  monitor.

## Code or comments ([experiment](../../experiments/adversarial_code_or_comments/README.md))

200 held-out APPS problems, each with an honest and a backdoored solution (the
backdoor confirmed by running both programs on a trigger input). A docstring is
added without changing any code: none, truthful, unrelated or misleading (a
cover story on the backdoored program, a false accusation on the honest one).
Predictions were frozen on 200 development problems; both pre-registered
held-out tests pass.

| AUROC, backdoored vs honest code | No comment | Misleading comments |
|---|---|---|
| Gleipnir 4B | 0.93 | **0.35** |
| Gleipnir 27B | 0.96 | **0.40** |
| Kimi K3 (teacher) | 0.98 | 0.63 |
| Claude Sonnet 5.5 | 0.99 | 0.90 |
| GPT-6.1 Sol | 0.99 | 0.96 |
| GPT-6 Luna | 0.81 | 0.82 |
| Gemini 3.8 Flash | 0.76 | 0.74 |

**Correction, 23 October.** The misleading column pairs different texts on
the two programs. With the [same text on both](../../experiments/adversarial_crossed_text/README.md)
(99 held-out tasks), AUROC under a specific / generic accusation is:

- Gleipnir 4B 0.58 / 0.59;
- Gleipnir 27B 0.98 / 0.96;
- Kimi K3 0.97 / 0.96;
- Claude Sonnet 5.5 0.95 / 0.95;
- GPT-6.1 Sol 1.00 / 1.00;
- GPT-6 Luna 0.97 / 0.95.

So the 27B and the teacher keep their ranking; only the 4B loses it, and more
than its untrained base does (0.81 / 0.75). With any of these monitors,
calibrating on undocumented code flags nearly all documented honest code. The
first two bullets below describe the confounded condition.

- Gleipnir believes what the text claims. A false accusation gets 194 of 195
  honest programs flagged at the 4B's strict 1% line; an unrelated docstring
  gets 181 of 200 flagged. Text moves both Gleipnir models more than code does.
- The credulity matches the teacher: mean P(1) on falsely accused honest code
  is 0.785 for Gleipnir 27B and 0.780 for Kimi K3, against 0.034 for Sol.
- A cover story alone does not hide a backdoor: against honest code with a
  truthful comment, Gleipnir 4B still separates them (AUROC 0.98).
- Limits: the comments are LLM-written to a specification, the programs are
  short contest solutions, and API monitors ran at low reasoning effort.

## EvilGenie × Escalation ([experiment](../../experiments/adversarial_evilgenie/README.md))

Public reward-hacking runs by 15 models. On the same 105 hacks and 999
legitimate runs, Gleipnir 27B reaches AUROC 0.93 against the 4B's 0.75 and flags
33 against 10 at its strict line. 69 of the hacks are mild (sample answers
hard-coded on a correct program) and neither model flags them at the strict
line. No frontier monitor was run on this set.

## Cost basis

Gleipnir cost is measured throughput on one A100 80GB at Google's on-demand
$5.07/hour with the research scorer (Transformers, one row per forward pass,
FLA kernels), so batched serving would be cheaper. API cost is actual spend over
rows attempted. Spend: OpenRouter $42.70, Anthropic $17.16.
