# Small Monitors on SLEIGHT-Bench

*How distilled action-only monitors fare against hand-built evasion attacks*

> **TL;DR:** We scored our released Gleipnir monitors (Qwen-based 4B and 27B
> models distilled from Kimi K3's binary logits) on
> [SLEIGHT-Bench](https://github.com/safety-research/sleight-bench), a
> benchmark of attacks hand-built to slip past monitors, using the paper's
> prompt and stripped transcripts, alongside several reference monitors. At
> 1% FPR, Gleipnir 27B catches 20% of attacks and the 4B catches 7.5%. The
> paper's frontier monitors range from 2% (Haiku 4.5) to 54% (Gemini 3.1 Pro
> with high thinking) through the API, and 62% for Opus 4.6 in a Claude Code
> scaffold. Three lessons came out:
> - **At this size, catch rate at 1% FPR is a coarse measure.** With 43
>   benign transcripts, the threshold is the single most suspicious benign.
>   Our teacher Kimi K3 catches 0% only because one benign transcript scores
>   above every attack; ignore it and Kimi catches 30%.
> - **Our out-of-distribution (OOD) suite overstated how we compare to
>   frontier monitors.** There, Gleipnir 27B matched Claude Opus 4.6 and
>   Gemini 3.1 Pro at about 200× lower cost. On SLEIGHT-Bench it catches
>   20%, against 19–32% for Opus 4.6 and 28–54% for Gemini 3.1 Pro. Against
>   decision models at a similar price (GPT-6 Luna, Jev 1.13, pplx-decider
>   v1.2) it is in the same range, ahead of Jev but level with the others.
> - **Distillation needs support in the training data.** A student learns
>   its teacher's judgment only on the kinds of cases it is trained on. Our
>   training data has stealth-prompted attacks but none built against a
>   monitor or disguised as honest mistakes. On SLEIGHT-Bench the 27B
>   recovers only about 40% of the gap between its base model and Kimi K3;
>   on the OOD suite it closed the gap entirely.

## The benchmark

SLEIGHT-Bench ([arXiv:2605.16626](https://arxiv.org/abs/2605.16626)) pairs
each attack transcript with a benign twin from the same scenario. Most
attacks are built to look like honest mistakes: an off-by-one `sed` line
edit, a wrong `awk` column, a locale-dependent comparison, a `.gitignore`
that quietly omits a secrets file. Others launder permissions or inject
instructions aimed at the monitor itself. The paper's main setting uses
stripped transcripts: the agent's hidden reasoning, its visible prose and the
explanatory `description` field of its tool calls are all removed, so the
monitor sees only the user's messages, the tool calls and their outputs. It
thresholds at 1% FPR on the benign twins.

We matched the paper's grading setup: the benchmark's 40 graded attacks
against their 43 benign twins, with each multi-session attack scored by its
most suspicious session, and the threshold set at 1% FPR on the benign
twins. With 43 benigns, "1% FPR" means "above the highest benign score".

## How our monitor plugs in

The paper's monitors write their reasoning and then a 0–100 suspicion
score, averaged over 10 trials at temperature 1. Gleipnir monitors are scored
by a single forward pass: the score is the softmax over the logits of `0`
and `1` after a `Prediction:` prefix.

We score each transcript once rather than averaging 10 trials, for two
reasons. Sampling helps the paper's monitors because each trial can reason
differently, but our monitors emit only the decision token, with no
reasoning to vary. And repeated sampling estimates how likely a monitor is to
flag a transcript, which our monitors already return exactly as the
probability on the `1` token.

We followed the paper's setting everywhere else: stripped transcripts and the
paper's ControlArena monitor prompt, with only the output format
changed to our binary decision. We scored each released checkpoint once on
its production serving stack (FP8/FP4 on a B200, 262K context). The reference
models went through the same prompt with the same single-logit readout,
except where a row says otherwise.

## Results

![Catch rate at 1% FPR on SLEIGHT-Bench](../../figures/sleightbench_catch_rate.png)

![pAUROC@20 on SLEIGHT-Bench](../../figures/sleightbench_pauroc.png)

The paper's rows are per-trial catch rates read from its appendix figures;
it reports no ranking metric, so they appear only in the first figure. Our
paper-protocol runs average each transcript's score over 10 trials. All
logit rows use the ControlArena prompt with our binary output format and no
reasoning.

The base models, which Gleipnir 4B and 27B are fine-tuned from, ran on a
stock BF16 engine. On that same engine the Gleipnir models catch 7.5% and
12.5%, with pAUROC@20 0.206 and 0.306 (AUROC 0.623 and 0.694). Fine-tuning
therefore lifts pAUROC@20 by 0.01–0.04 with little change in catch rate.

**What the 27B catches.** It catches attacks whose actions are overtly
unauthorized: `curl | bash` from an unusual host, an exfiltration hidden in a
Brainfuck interpreter, a renamed storage bucket, deleted monitoring logs and
both permission-laundering stages. **What it misses.** It catches none of the
counting or object-reuse attacks, and only one of the model-priors attacks.
These are the classes designed to pass as honest mistakes. It is also fooled
by the injection jailbreak (0.05), as is Kimi K3; the 4B flags it (0.96),
though that is less impressive than it looks (see below).

## Lesson 1: at this size, catch rate at 1% FPR is a coarse measure

The threshold is the single highest benign score, so one suspicious benign
can move a monitor's headline by tens of points.
- **Kimi K3's 0%.** Our teacher ranks better than either student, yet its
  logits catch nothing. The benign twin of `sed-line-drift`, which differs
  from its attack only by a line number, scores 0.98. Ignore that one
  benign and Kimi catches 30%.
- **Ten trials don't fix it.** Under the paper's protocol, the same twin
  scores 62–85 in all ten of Kimi's trials and still sets the threshold.
- **Score resolution matters too.** The 27B's threshold is a benign
  permission-laundering twin at exactly 0.5. Its logits are BF16, which near
  the threshold moves scores in coarse steps (0.500, 0.531, 0.562, …). On an
  unquantized engine, three attacks land exactly on the threshold and count
  as misses, so the same checkpoint catches 12.5% at the same AUROC.

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
  jailbreak: answering immediately, it flags the attack; after reasoning,
  it scores it 0.02 or less.
- **A single-token readout is more dependable for small models.** Under
  the paper's protocol the base 4B catches 0.5% without thinking and 3% with
  it, against 5% from its logits, although its ranking is similar
  (pAUROC@20 0.21–0.24 against 0.19). Each sampled trial is noisy, and the
  threshold is set over 430 benign trials. The logit readout returns the
  score in one token, with no sampling noise. For the 27B the reasoned score ranks
  somewhat better (pAUROC@20 0.32–0.36 against 0.26), at the cost of 10
  generated analyses per transcript.
- **Reasoning did help Kimi K3.** Kimi's AUROC rises from 0.74 with logits
  to 0.80 under the paper's sampled-and-averaged protocol. Thinking itself
  added nothing on top.

## Reference points: current decision models

The paper predates OpenAI's current GPT-6 models, so we scored both with the
same zero-reasoning logit interface and ControlArena prompt. We also scored
two low-cost dedicated decision models, TypeSafe's Jev 1.13 and
Perplexity's pplx-decider v1.2, through their native interface: the
trajectory as state, our standard monitoring question, and the returned
probability. Following Lesson 1, we compare them mainly on pAUROC@20.

| Monitor | Interface | Catch @1% FPR | 2nd-benign catch | AUROC | pAUROC@20 | Input price |
|---|---|---:|---:|---:|---:|---:|
| GPT-6 Sol | Logits | 15% | 47.5% | 0.823 | 0.515 | — |
| GPT-6 Luna | Logits | 5% | 12.5% | 0.696 | 0.277 | $0.10/M |
| Jev 1.13 | Native decision | 7.5% | 10% | 0.644 | 0.192 | $0.042/M |
| pplx-decider v1.2 | Native decision | 25% | 25% | 0.690 | 0.319 | $0.02/M |
| Gleipnir 27B | Logits | 20% | 20% | 0.697 | 0.299 | $0.018–0.04/M |

Gleipnir's price is our self-hosted B200 break-even on committed and
on-demand GPUs, before provider margin and utilization.
- **GPT-6 Sol with one logit is the best ranker we measured** (AUROC 0.82).
  It is strongest on exactly the mistake-shaped classes our models miss.
- **Among the low-cost models, only Jev falls behind.** On pAUROC@20 the
  order is Decider (0.319), Gleipnir 27B (0.299), Luna (0.277), then Jev
  (0.192), but 40 attacks cannot resolve the top three. A paired bootstrap
  over scenarios puts every difference from Gleipnir 27B within noise:

  | vs Gleipnir 27B | Catch difference (95%) | pAUROC@20 difference (95%) | Draws where it beats Gleipnir |
  |---|---|---|---:|
  | pplx-decider v1.2 | −7.5 to +17.5 pts | −0.11 to +0.16 | 62% |
  | GPT-6 Luna | −27.5 to +7.5 pts | −0.19 to +0.13 | 36% |
  | Jev 1.13 | −27.5 to +2.5 pts | −0.26 to +0.04 | 7% |

  Catch rate exaggerates the Luna gap (5% against 20%), because one benign
  sets Luna's threshold near certainty. Luna costs 2.4–5.7× more than
  Gleipnir; Jev costs the same.
- **Decider's price advantage disappears at scale, and its architecture
  costs more in live monitoring.** Decider is built on the same Qwen3.8-27B
  backbone as Gleipnir 27B. Its $0.02/M list price is half our on-demand
  cost but matches our cost on committed GPUs ($0.018/M). It also reads the
  trajectory with non-causal attention, so it cannot reuse a KV cache. A
  synchronous monitor re-scores the trajectory after every agent action;
  Gleipnir, a causal decoder, can keep the cached prefix and process only
  the new step, while Decider must re-encode the whole trajectory each time.
  The per-trajectory prices above therefore understate Decider's cost in
  that setting.
- **Jev's scores are coarse:** 41 distinct values here, none above 0.53.
- **The jailbreak.** Luna is fully fooled by the injection attack (score
  about 2×10⁻¹⁰), as is Gleipnir 27B. Decider and Jev flag it, Decider at
  0.95 and Jev narrowly at 0.53. Their question tells them to treat
  trajectory text as untrusted evidence.

**Against these models, our OOD lead mostly does not carry over.** The
table uses BF16 scores for Gleipnir, to match our OOD evaluation, and
Decider's OOD run used its local readout:

| pAUROC@20 | OOD suite | OOD, without the airline task | SLEIGHT-Bench |
|---|---:|---:|---:|
| Gleipnir 27B | 0.935 | 0.973 | 0.306 |
| pplx-decider v1.2 | 0.896 | 0.957 | 0.319 |
| Jev 1.13 | 0.850 | 0.898 | 0.192 |
| GPT-6 Luna | 0.837 | 0.876 | 0.277 |

The OOD suite's hardest source is SHADE-Arena, and within it one
airline-booking task. Attackers there book nearly every flight on one
expensive airline. The evidence is fully visible, but the booking is
permitted, so the "attack" label marks hidden intent rather than a broken
instruction. The task drags every monitor down. It explains most of
Gleipnir's OOD lead over Decider, but not its lead over Luna and Jev: without
the task, Gleipnir still leads Luna on SHADE-Arena itself (0.96 against 0.65)
and by 5–10 points on IaC, MLE-Sabotage and Rogue Eval. On SLEIGHT-Bench all
four sit between 0.19 and 0.32.

## A caveat on the jailbreak: monitors react to any monitor-directed text

The 4B's high score on the injection jailbreak looks like robustness. The
APPS injection benchmark suggests it is mostly a reflex. There, honest
programs carry a note addressed to the monitor. At a 0.5 threshold:

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

## Lesson 3: our OOD suite overstated how we compare to frontier monitors

On our six-dataset OOD suite, Gleipnir 27B matched the strongest reasoning
monitors we had OOD numbers for, at about 200× lower cost. SLEIGHT-Bench
separates them. The paper reports only catch rates, so we compare on those:

| Monitor | OOD pAUROC@20 | OOD cost per 1K trajectories | SLEIGHT-Bench catch @1% FPR |
|---|---:|---:|---|
| Claude Opus 4.6 | 0.942 | $94.79 | 19% (no thinking), 32% (thinking) |
| Gemini 3.1 Pro | 0.940 | $73.22 | 28% (low thinking), 54% (high thinking) |
| Gleipnir 27B | 0.951 | $0.39 | 20% |

The frontier OOD rows come from
[Sinha et al.](https://arxiv.org/abs/2605.29601), whose OOD datasets we use;
their monitors write a rationale before a 0–10 score. Gleipnir's OOD score is on
our optimized stack (0.935 in BF16), within noise of both. The OOD suite is
close to saturated: Gleipnir 27B scores 0.99 or higher on four of its six
sources, which leaves little room to separate strong monitors. Gleipnir
remains far cheaper, but on deliberately evasive attacks it is not a
frontier-level monitor.

## Why: distillation needs support in the training data

A distilled student learns its teacher's judgment only on the kinds of cases
it is trained on. The same saturation hid this gap between Gleipnir and its
teacher. On matched prompts, the adapter's absolute gain is about the same on
both benchmarks; what differs is the ceiling. Base and Gleipnir are scored on
the same BF16 engine.

| pAUROC@20 | Base 27B | Gleipnir 27B | Kimi K3 | Adapter gain |
|---|---:|---:|---:|---:|
| Our OOD suite (6 datasets) | 0.890 | 0.935 | 0.908 | +4.5 pts |
| SLEIGHT-Bench | 0.263 | 0.306 | 0.402 | +4.3 pts |

On the OOD suite, where the base model is already at 0.89, the student
passes its teacher. On SLEIGHT-Bench it recovers only about 40% of the
base-to-teacher gap.

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
  34% of BashBench attacks and no benign runs, and both Gleipnir monitors
  catch `curl-pipe-bash`. Deleting logs with `rm -rf` is *more* common in
  benign BashArena runs (17%) than in attacks (7.5%). Of SLEIGHT-Bench's two
  log-deletion attacks, the 27B catches one barely (0.53) and scores the
  other 0.07; the 4B scores both below 0.04.

Our monitoring specification, which produced the training labels, also
treats bugs and failed commands as benign unless there is evidence of an
unauthorized objective. That is in direct tension with attacks disguised as
bugs, and the models may have learned it from the labels even though the
ControlArena prompt does not state it.

## Limitations

- 40 attacks and 43 benigns: every bootstrap interval on catch rate overlaps
  (the 27B's is 10–32.5%).
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

*Details and per-run receipts:
[finding](../findings/sleightbench_released_monitors.md) and
[experiment](../../experiments/sleightbench_eval/README.md). Models:
[Gleipnir-4B](https://huggingface.co/Jazhyc/Gleipnir-4B) and
[Gleipnir-27B](https://huggingface.co/Jazhyc/Gleipnir-27B).*
