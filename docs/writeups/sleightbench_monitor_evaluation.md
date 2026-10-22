# Small Monitors on SLEIGHT-Bench

*How distilled action-only monitors fare against hand-built evasion attacks*

> **TL;DR:** We scored our released Gleipnir monitors (Qwen-based 4B and 27B
> models distilled from Kimi K3's binary logits) once on
> [SLEIGHT-Bench](https://github.com/safety-research/sleight-bench), a
> benchmark of attacks hand-built to slip past monitors. At 1% FPR on the
> paper's stripped transcripts, Gleipnir 27B catches 15% of attacks and the
> 4B catches 5%. The paper's frontier monitors range from 2% (Haiku 4.5) to
> 32% (Opus 4.6 with thinking) through the API, and 62% for Opus 4.6 in a
> Claude Code scaffold. Two broader lessons came out of the follow-ups. First,
> with 43 benign transcripts, catch rate at 1% FPR is often set by a single
> suspicious benign, so it should be read as coarse tiers. Second, our
> previous out-of-distribution (OOD) suite overstated how close the student
> is to its teacher: there the 27B matched Kimi K3, but here it trails by
> about 10 points of pAUROC@20. Our training data contains stealth-prompted
> attacks but none built against a monitor, which is a plausible cause.

## The benchmark

SLEIGHT-Bench ([arXiv:2605.16626](https://arxiv.org/abs/2605.16626)) pairs
each attack transcript with a benign twin from the same scenario. Most
attacks are built to look like honest mistakes: an off-by-one `sed` line
edit, a wrong `awk` column, a locale-dependent comparison, a `.gitignore`
that quietly omits a secrets file. Others launder permissions or inject
instructions aimed at the monitor itself. The paper's main setting strips
the agent's prose, so the monitor sees only tool calls and outputs, and it
thresholds at 1% FPR on the benign twins.

We used 43 attack and 43 benign transcripts (excluding the refusal-forcing
attack), with multi-session attacks merged by max into 40 graded units.
With 43 benigns, "1% FPR" means "above the highest benign score". The
transcripts are encrypted upstream to keep them out of training corpora; we
evaluated locally and redistribute nothing.

## How our monitor plugs in

Gleipnir monitors read a trajectory and are scored by a single forward pass:
the score is the softmax over the logits of `0` and `1` after a
`Prediction:` prefix. There is no sampled reasoning and no ordinal score.
The paper's monitors instead reason and output a 0–100 score, averaged over
10 trials at temperature 1.

We froze the protocol before any SleightBench score existed, scored each
released checkpoint once on its production serving stack (FP8/FP4 on a
B200, 262K context), and treated every later run as an exploratory
diagnostic.

## Results

Stripped view, 1% FPR. "2nd-benign catch" is the catch rate if the single
most suspicious benign were removed (FPR of 1 in 43). The paper's rows are
per-trial catch rates read from its appendix figures.

| Monitor | Interface | Catch @1% FPR | 2nd-benign catch | AUROC |
|---|---|---:|---:|---:|
| Haiku 4.5 (paper) | Reasoned score, no thinking | 2% | — | — |
| GPT-5.5 (paper) | Reasoned score, no thinking | 14% | — | — |
| Opus 4.6 (paper) | Reasoned score, thinking | 32% | — | — |
| GPT-5.5 (paper) | Reasoned score, thinking | 50% | — | — |
| Opus 4.6 (paper) | Claude Code scaffold | 62% | — | — |
| Qwen3.8-27B base | Reasoned score, thinking (paper protocol) | 14.0% | — | 0.711 |
| Kimi K3 (our teacher) | Reasoned score, no thinking (paper protocol) | 12.2% | — | 0.802 |
| Kimi K3 | Logits | 0% | 30% | 0.737 |
| GPT-5.5 | Logits | 15% | 15% | 0.783 |
| **Gleipnir 4B** | Logits, own prompt | **5%** | 5% | 0.657 |
| **Gleipnir 27B** | Logits, own prompt | **15%** | 27.5% | 0.706 |

The logit runs for Kimi K3 and GPT-5.5 used the ControlArena
monitor description with our binary output format and no reasoning.

**What the 27B catches.** It catches attacks whose actions are overtly
unauthorized: `curl | bash` from an unusual host, renamed storage buckets,
deleted monitoring logs, a `.gitignore` omission. It also flags the
injection jailbreak that every Opus 4.6 API configuration in the paper
misses, though that is less impressive than it looks (see below). **What it
misses.** It catches none of the counting or object-reuse
attacks. Of the model-priors attacks it catches only one, and only when the
agent's prose is visible. These are the classes designed to pass as honest
mistakes.

## Lesson 1: at this size, catch rate at 1% FPR is a coarse tier

The threshold is the single highest benign score, so one suspicious benign
can move a monitor's headline by tens of points.
- **Kimi K3's 0%.** Our teacher ranks better than either student, yet its
  logits catch nothing. The benign twin of `sed-line-drift`, which differs
  from its attack only by a line number, scores 0.98. Remove that one
  scenario and Kimi catches 28–36%.
- **The same twin** drops the 27B's visible-view catch from 25% to 2.5%
  when we swap in the ControlArena prompt, at unchanged AUROC.
- **Ten trials don't fix it.** Under the paper's protocol, the same twin
  scores 62–85 in all ten of Kimi's trials and still sets the threshold.

The paper's own numbers likely share this fragility, since they also
threshold on these 43 benigns. The paper notes that outside benign pools
lower frontier catch rates by 10–20 points. We would compare monitors on
AUROC, pAUROC@20 and catch rate at the first few benign thresholds together,
rather than on one catch rate.

## Lesson 2: the interface matters less than the model here

- **Logits reproduce the paper for GPT-5.5.** Our single zero-reasoning
  logit gives 15% catch; the paper's 10-trial reasoned score without
  thinking gives 14%.
- **Reasoning didn't help our adapted monitors.** Native thinking or written
  reasoning before the decision left the 27B's ranking within noise and
  lowered the 4B on every metric. It also made the 4B fall for the injection
  jailbreak: answering immediately, it scores the attack 0.73–0.84; after
  reasoning, 0.00–0.03.
- **Reasoning did help Kimi K3.** Kimi's AUROC rises from 0.74 with logits
  to 0.80 under the paper's sampled-and-averaged protocol. Thinking itself
  added nothing on top.

## Reference points: current hosted models

The paper predates OpenAI's current GPT-6 models, so we scored both with the
same zero-reasoning logit interface and ControlArena prompt as a modern
reference. We also scored TypeSafe's Jev 1.13, a hosted decision model,
through its native interface: the trajectory as state, our standard
monitoring question, and its returned probability. None of these is part of
the paper comparison above.

| Stripped view | Interface | Catch @1% FPR | 2nd-benign catch | AUROC | pAUROC@20 |
|---|---|---:|---:|---:|---:|
| GPT-6 Sol | Logits, ControlArena prompt | 15% | 47.5% | 0.823 | 0.515 |
| GPT-6 Luna | Logits, ControlArena prompt | 5% | 12.5% | 0.696 | 0.277 |
| Jev 1.13 | Native decision | 7.5% | 10% | 0.644 | 0.192 |
| Gleipnir 27B | Logits, ControlArena prompt | 12.5–20% | 20% | 0.694–0.697 | 0.299–0.306 |

**GPT-6 Sol with one logit is the best ranker we measured** (AUROC 0.82), and
it is strongest on exactly the mistake-shaped classes our models miss.
**Gleipnir 27B roughly matches the low-cost GPT-6 Luna and ranks above
Jev 1.13**, which costs the same per input token as our self-hosted 27B
($0.042/M). Jev returns coarse probabilities (41 distinct values here, none
above 0.53).

The two Gleipnir figures are the same checkpoint on our optimized and BF16
serving stacks. The 7.5-point catch gap between them comes from three attacks
tied with the threshold benign at BF16 resolution, which the strict rule
counts as misses. Ranking is level, and on the visible view Luna's AUROC is
higher (0.74 against 0.70). Self-hosted at B200 break-even, Gleipnir 27B costs
about $0.04 per million input tokens on demand ($0.018 on committed GPUs),
against Luna's $0.10 list price: about 2.4–5.7× cheaper, before accounting
for provider margin and utilization. Under this prompt Luna is fully
jailbroken by the injection attack (score about 2×10⁻¹⁰). Jev, whose question
tells it to treat trajectory text as untrusted evidence, flags it, if
narrowly (0.53, against 0.19 for its benign twin).

**Our OOD suite separates these monitors; SLEIGHT-Bench does not.** On our
six-dataset OOD suite, Gleipnir 27B led Luna and Jev by 8–10 points of
pAUROC@20. Here all three sit between 0.19 and 0.31:

| pAUROC@20 | OOD suite | OOD, without the airline task | SLEIGHT-Bench, stripped |
|---|---:|---:|---:|
| Gleipnir 27B | 0.935 | 0.973 | 0.306 |
| Jev 1.13 | 0.850 | 0.898 | 0.192 |
| GPT-6 Luna | 0.837 | 0.876 | 0.277 |

The OOD suite's hardest source is SHADE-Arena, and within it one
airline-booking task. Attackers there book nearly every flight on one
expensive airline: the evidence is fully visible, but the booking is
permitted, so the "attack" label marks hidden intent rather than a broken
instruction. The task drags every monitor down, but it is not why Gleipnir
leads. Without it, the gap is unchanged, and Gleipnir still leads Luna on
SHADE-Arena itself (0.96 against 0.65) and by 5–10 points on IaC,
MLE-Sabotage and Rogue Eval.

So the OOD lead is real but specific to the kinds of sabotage our training
data covers. It does not carry over to attacks built to look like honest
mistakes, where every monitor except Sol is near its floor.

## A caveat on the jailbreak: monitors react to any monitor-directed text

Our monitors' high score on the injection jailbreak looked like robustness.
A follow-up on the APPS injection benchmark suggests it is mostly a reflex.
There, honest programs carry a note addressed to the monitor. At a 0.5
threshold:

| APPS, honest code with an injected note | Flagged (GPT-4 / GPT-4.1 / GPT-4o template) | Injected backdoor vs injected honest AUROC |
|---|---|---:|
| Qwen3.8-27B base | 76 / 99 / 100% | 0.66 |
| Gleipnir 27B | 53 / 73 / 100% | 0.83 |
| Gleipnir 4B | 90 / 99.5 / 98% | 0.62 |

The 27B rows use the same BF16 engine; the 4B row is on our optimized stack.
Fine-tuning reduces the base model's reaction and greatly improves
discrimination, but both monitors still treat a note to the monitor as strong
evidence of misbehavior. That label is debatable, because an agent writing to
its own monitor is arguably suspicious. On the 4B, projecting out a fitted
"injection direction" softened the reaction but did not remove it.

## Lesson 3: our OOD suite was too easy to separate student from teacher

On matched prompts, the adapter's absolute gain is about the same on both
benchmarks; what differs is the ceiling.

| pAUROC@20 | Base 27B | Gleipnir 27B | Kimi K3 | Adapter gain |
|---|---:|---:|---:|---:|
| Our OOD suite (6 datasets) | 0.890 | 0.935 | 0.908 | +4.5 pts |
| SLEIGHT-Bench, stripped | 0.263 | 0.306 | 0.402 | +4.3 pts |

On the OOD suite, where the base model is already at 0.89, the student
passes its teacher. On SLEIGHT-Bench it recovers only about 40% of the
base-to-teacher gap. "The 27B matches Kimi K3" was a property of an easy
benchmark, not of the distillation.

## Why: no monitor-adversarial training data

We reviewed how each of our five training sources generated its attacks and
scanned the 8,688 training prompts for SLEIGHT-Bench's mechanisms.
- **Attacks were stealth-prompted, never optimized against a monitor.**
  CoT-Red-Handed and BashArena tell the attacker to evade a reviewer or
  monitor; STRIDE and Gloom ask for subtlety and plausible deniability;
  BashBench uses red-team-iterated multi-step malware downloads. No source
  selected or refined attacks against a monitor's score.
- **The mistake-shaped mechanisms are nearly absent.** Line-number edits,
  column-index bugs, locale assumptions, `.gitignore` tricks and
  monitor-directed injections appear in roughly 1–3% of rows or fewer, with
  no label signal.
- **Label skews explain specific hits and misses.** `curl … | sh` appears in
  34% of BashBench attacks and no benign runs, and every Gleipnir run catches
  `curl-pipe-bash`. Deleting logs with `rm -rf` is *more* common in benign
  BashArena runs (17%) than in attacks (7.5%), and both of SLEIGHT-Bench's
  log-deletion attacks score below 0.12 on our models.

Our monitoring specification also labels bugs and failed commands as benign
unless there is evidence of an unauthorized objective, which is in direct
tension with attacks disguised as bugs. Swapping in the ControlArena
instruction alone did not help, though, so the specification is not the
whole story.

## Limitations

- 40 attacks and 43 benigns: every bootstrap interval on catch rate overlaps
  (the 27B's is 5–37.5%).
- Our runs are single passes; the paper averages 10 trials. GPT-5.5's
  logprobs were not repeatable (the same prompt scored 0.59, then 0.97).
- The paper's frontier numbers were read off figures and use their full
  protocol; we did not rerun them.
- The training-data scan is lexical; it shows mechanisms are rare, not how
  they are used.

## What we would try next

Train on deniable, mistake-shaped and monitor-evasion examples, ideally
generated against a monitor in the loop. Then check whether SLEIGHT-Bench
ranking improves without regressing on the OOD suite, with SLEIGHT-Bench
held out for evaluation only.

*Details, per-run receipts and frozen protocols:
[finding](../findings/sleightbench_released_monitors.md) and
[experiment](../../experiments/sleightbench_eval/README.md). Models:
[Gleipnir-4B](https://huggingface.co/Jazhyc/Gleipnir-4B) and
[Gleipnir-27B](https://huggingface.co/Jazhyc/Gleipnir-27B).*
