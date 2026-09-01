# Control Monitoring Has an Interface Problem

## Simple logit distillation for stronger, more general monitors

> Status: Working draft. This Markdown file is the canonical source for the
> project writeup until it is exported to Google Docs.

## 1. Background: training control monitors

### The monitoring problem

The [Hugging Face
incident](https://openai.com/index/hugging-face-incident-and-the-road-ahead/)
made the need for monitoring concrete: OpenAI agents escaped isolation,
coordinated through unauthorized channels, and compromised external and
internal infrastructure while monitors that retrospectively could have raised
an early alert were not running. It also exposed a practical constraint.
Frontier labs are under pressure to improve capabilities, while monitoring adds
compute and operational overhead to development. Rather than accept safeguards
being narrowed or omitted, we should make monitoring strong and economical
enough to run by default.

We focus on **action-only monitoring**, where the monitor sees messages, tool
calls, commands, outputs, edits, and the final report rather than private chain
of thought or model internals. Raw chain of thought may be unavailable or
unreliable, while observable actions can be logged consistently. This setting
also has established benchmarks and methods that let us make useful comparisons
within a short project.

### The previous cost--performance frontier

Our main reference is Sinha et al.'s [*Training Deliberative Monitors for
Black-Box Scheming Detection*](https://arxiv.org/abs/2605.29601). To our
knowledge, it is the first paper to establish a cost--performance frontier for
action-only control monitors while also training smaller open-weight models to
improve that frontier. The paper provides a concrete training pipeline, a
six-dataset OOD evaluation suite, and a shared cost and performance framework,
making it the clearest baseline for our project.

Sinha et al. give the monitor a detailed specification of problematic
behavior. The specification tells it to identify relevant actions, compare the
trajectory with the agent's final report, consider an innocent explanation,
consider a misaligned explanation, and reach a calibrated conclusion. The
monitor then writes a structured rationale and returns an integer from 0 to 10.
Low scores represent task-serving behavior, middle scores represent ambiguity,
and high scores represent clear or deliberately concealed misalignment.

Their training process distils this deliberation into smaller models. For each
training example, Gemini 2.5 Pro generates four candidate rationales. Candidates
that point in the wrong binary direction can be regenerated with the known
label, and Claude Sonnet 4.5 scores the remaining rationales for grounding,
two-sided analysis, calibration, and consistency. The best acceptable rationale
becomes the supervised fine-tuning target. The paper then optionally continues
training with reinforcement learning. This produces monitors that have been
trained to generate both the analysis and the final ordinal score.

The training mixture contains 8,688 balanced trajectories from five sources.
Evaluation uses 6,395 examples from six separate sources covering computer-use
sabotage, infrastructure changes, prompt-injection compliance, ML sabotage,
oversight tampering, and multi-tool exfiltration. These evaluation sources are
treated as strict OOD data: they are not used to choose prompts, checkpoints,
thresholds, or stopping rules.

The generated-rationale interface is also practical in a field dominated by
model APIs. Closed frontier models reliably return text, but providers do not
consistently expose the logits needed to score a fixed decision boundary. Even
when some log probabilities are available, the supported tokens and request
formats differ across providers. A rationale followed by an integer score can
therefore be used across many closed and open models without relying on a
provider-specific logit interface. We view this as an important reason to take
the paper's design seriously, even though we test whether it is the best
interface when logits are available.

### The evaluation metric

Each monitor assigns one score to every trajectory. AUROC evaluates the ranking
created by those scores. It can be understood as the probability that a random
positive example receives a higher score than a random negative example, with
ties handled explicitly. The ROC curve shows the true-positive and
false-positive rates obtained as the decision threshold moves across the score
distribution. AUROC measures the area under the full curve and does not require
choosing one threshold in advance.

Sinha et al. use **pAUROC@20** as their primary metric. It measures only the
part of the ROC curve where the false-positive rate is at most 20%, then divides
the area by 0.2 so the result remains on a zero-to-one scale:

```text
pAUROC@20 = (1 / 0.2) * integral from FPR=0 to 0.2 of TPR(FPR) dFPR
```

This metric gives more weight to the operating region that matters for
oversight. A detector can have a good full AUROC while still ranking too many
benign examples near the top. pAUROC@20 asks whether the monitor can recover
positive cases before false alarms become common.

The paper computes pAUROC@20 independently on each of the six OOD datasets and
reports their unweighted mean. This prevents a large dataset from dominating
the result and makes each task family count equally. We retain this exact
aggregation. We also retain the paper's cost coordinate: uncached,
token-metered inference cost per 1,000 evaluations. Together, the two numbers
measure how well a monitor ranks harmful behavior in the low-false-positive
region and how much it costs to run.

### Our starting point: Aletheia's Quest

Gleipnir did not begin from a blank slate. The project carries over hypotheses
and infrastructure developed during [Aletheia's
Quest](https://aletheias-quest.github.io/), a competition on detecting deception
across multiple language-model organisms and datasets. Our methods reached
first place on the public leaderboard.

The main lesson from the competition was that a teacher's decision boundary is
a useful training signal. Instead of asking only for a hard label, we collected
the teacher's relative logits for literal binary decision tokens and trained
Qwen students to match the resulting probability distribution. We combined
this objective with QLoRA, continuous decision scores, grouped evaluation, and
reproducible caches that record the teacher, prompt, token usage, and raw
log-probability response.

The competition also gave us a working training and systems recipe. We had
already tested teacher collection, adapter capacity, data scaling, inference
parity, and large evaluation runs. Gleipnir is therefore mainly a rapid transfer
project: move the highest-value ideas from deception detection to action-only
control monitoring, preserve Sinha et al.'s evaluation contract, and find out
which gains survive the change in domain.

### Why build a strong black-box baseline?

The goal of this project is to build the strongest practical monitoring
baseline we can from observable behavior alone. This is useful even if
black-box monitoring cannot solve the whole alignment problem.

Interpretability research often values methods because they expose a new object
of study or offer a new way to describe a model. That scientific value is real,
but it is different from showing that the method improves monitoring. Sparse
Autoencoders are a useful example. They may produce representations that are
easier to inspect, but that does not by itself show that they detect scheming or
sabotage better than a strong monitor trained on model outputs. If the black-box
baseline is weak, an apparent white-box gain may come from better training,
better labels, or a better decision interface rather than privileged access to
the model's internals.

At the same time, output monitoring has a hard limit. Two models can produce
the same visible behavior while differing in their hidden goals or
computations. A model may also learn to conceal the evidence that a black-box
monitor relies on. White-box methods have a better chance in these cases
because activations and other internal signals may contain information that is
absent from the observable trajectory.

That makes a strong black-box baseline an important experimental control. A
future white-box monitor should be compared with it under matched training
data, evaluation data, and compute. If the white-box method still wins, we have
better evidence that the gain comes from privileged internal information. By
making output-only monitoring as strong and simple as possible, Gleipnir raises
that bar and makes the contribution of future interpretability methods easier
to identify.

## 2. The interface problem

### Severity is not confidence

- Explain that a generated 0--10 score asks the model to combine two different
  concepts: how severe an action would be and how confident the monitor is that
  the action occurred.
- Show why the mapping is unclear. An uncertain but severe attack and a
  confident but moderate violation can plausibly receive the same ordinal
  score even though a binary detector should rank them differently.
- Contrast this with a binary interface. The normalized logits for literal
  `0` and `1` directly define the model's relative confidence that the
  trajectory is problematic.
- Connect the interface to AUROC and pAUROC@20: both metrics need a continuous
  ranking score, so binary decision logits expose the quantity the evaluator
  actually uses.

### Matched Qwen3.5-9B ablations

- Compare four frozen conditions on the same model and OOD suite:
  - immediate binary logits: `0.6976` Mean-OOD pAUROC@20;
  - immediate ordinal score: `0.4858`;
  - visible reasoning followed by binary logits: `0.6923`;
  - visible reasoning followed by an ordinal score: `0.6656`.
- Emphasize the main result: visible reasoning recovers much of the ordinal
  interface's loss, but neither reasoned condition beats immediate binary
  logits on the primary metric.
- Note that immediate binary logits are also the cheapest of the four
  conditions.
- Use the quantization control to rule out a simple explanation based on score
  resolution: rounding binary probabilities to 11 values retains `0.6835`
  pAUROC@20, while the model's generated ordinal scores reach only `0.4858`.
- Keep the conclusion narrow: this tests one Qwen3.5-9B prompting setup and
  does not show that reasoning is universally harmful.

### Results at larger scales

- Show the Qwen3.5-4B, Qwen3.5-27B, and Kimi K2.6 binary-logit results to
  establish how the interface behaves across model scales.
- Highlight Qwen3.5-27B: immediate binary logits reach `0.8395` pAUROC@20 at
  `$1.8476` per 1,000 evaluations, compared with `0.831` and `$2.775` for the
  paper's trained Qwen3.5-27B SFT+RL monitor.
- Highlight Kimi K2.6: immediate binary logits reach `0.9194` at `$7.9110`,
  compared with `0.841` and `$10.175` for the paper's Kimi K2.6 SFT monitor.
- Note that the 27B and Kimi comparisons support the interface hypothesis but
  are not matched causal ablations: the prompt, training, and output procedure
  also differ from the paper's conditions.
- Planned evidence: a compact table comparing the 9B interface ablations and
  the larger binary-logit results.

## 3. Simple soft distillation

### Replacing rationale distillation with decision distillation

- Use the existing pipeline diagram to compare the two methods.
- Sinha et al.: sample several long rationales per example, judge them with a
  second frontier model, select or regenerate candidates, train on the chosen
  rationale and score, and optionally run RL.
- Gleipnir: ask Kimi K3 for the binary decision-token distribution, cache the
  two logits, and train the student to match that distribution.
- Explain what is removed: no generated teacher rationale is stored as a
  target, no rationale judge or candidate-selection stage is required, and the
  student is not trained to reproduce an explanation.
- Explain what is retained: the teacher receives the full monitoring
  specification, while the deployed student uses the paper's compact task
  instruction.

### Training Gleipnir 4B and 9B

- Train Qwen3.5-4B and Qwen3.5-9B rank-128 QLoRA adapters using only Kimi K3's
  soft binary targets.
- State the transferred competition recipe: learning rate `5e-5`, effective
  batch size 32, and direct soft-target supervision.
- State the deliberate speed choice: one epoch rather than a tuned multi-epoch
  schedule.
- Describe the full mixed training set: 8,688 matched action-only trajectories
  plus 13,149 deception examples from Aletheia's Quest.
- Make clear that all model and prompt choices were frozen before scoring the
  six-source OOD suite.

### A new cost--performance frontier

- Introduce the frontier as point-estimate nondominance in Mean-OOD pAUROC@20
  and uncached inference cost per 1,000 evaluations.
- Gleipnir 4B reaches `0.7823` pAUROC@20 at `$0.2598` per 1,000 evaluations. It
  is the lowest-cost observed frontier point and improves over its unadapted
  4B baseline by `0.1649`.
- Gleipnir 9B reaches `0.8212` at `$0.8659`. It is the next Gleipnir frontier
  point and approaches the unadapted 27B result at less than half the cost.
- Compare these points with the paper's trained monitors while keeping the
  shared metric and cost convention explicit.
- Planned figure: the existing OOD cost--performance frontier, with the base
  models, Gleipnir adapters, and paper reference points.
- State the scope: these are research LoRA adapters and cost comparisons, not
  production-ready safety systems.

## 4. What does soft distillation teach?

### Exploratory explanation-quality analysis

- Motivate the question: soft distillation never supplies a rationale target,
  so does it improve only the final score or also the model's visible analysis?
- Describe the frozen 64-pair exploratory comparison between Gleipnir 4B and
  its unadapted backbone using a blinded GPT-5.6 Sol judge.
- Report the main result: mean overall explanation-and-decision score rises
  from `6.391` to `7.156`, a paired difference of `+0.766` with a bootstrap
  95% interval of `[+0.109, +1.453]`.
- Separate decision quality from explanation quality. On pairs where both
  models make the same prediction, the estimated gain is only `+0.240` and its
  interval crosses zero.
- State the supported conclusion: soft distillation improves the complete
  visible analysis-and-decision output on this sample, but independent
  reasoning improvement remains inconclusive.
- Add the main caveats: the judge score gives only a relative ordering, it does
  not test hidden-reasoning faithfulness, and Sinha et al. do not report a
  directly comparable metric for the reasoning quality of their monitors.

### Data scaling is useful but not smooth

- Present the one-seed Qwen3.5-9B curve over 204, 504, 996, 2,004, 4,008, and
  8,688 matched training examples.
- Highlight the transition around 996 examples: Mean-OOD AUROC rises from
  `0.8846` at 504 rows to `0.9353` at 996 rows.
- Highlight the non-monotonic result: pAUROC@20 peaks at `0.8365` with 4,008
  rows and falls to `0.8227` with 8,688 rows.
- Compare the curve descriptively with Sinha et al.'s digitized Qwen3.5-4B and
  27B rationale-SFT curves. Gleipnir 9B shows a larger gain and exceeds the
  paper's 4B curve at every matched row count.
- Do not make a causal claim from this comparison: the base models, number of
  epochs, subset construction, checkpoint selection, targets, and interfaces
  differ.
- Planned figure: the existing OOD data-scaling plot with the paper's curves
  shown as approximate references.

### Limits of cross-domain transfer

- Compare the 8,688-row action-only adapter with the 21,837-row mixed adapter
  that adds 13,149 Aletheia's Quest deception examples.
- Report the source-balanced result: Mean-OOD pAUROC@20 changes from `0.8227`
  to `0.8212`, while Mean-OOD AUROC is effectively unchanged.
- Note that pooled metrics and individual sources move differently, so the
  result is not evidence that every example or domain is unaffected.
- State the practical interpretation: adding familiar deception data did not
  produce noticeable aggregate transfer to the six action-only OOD domains.
- Keep the conclusion provisional because the campaign uses one seed, one
  epoch, and inherited rather than newly optimized hyperparameters.
- End with the main boundary: a better interface and simple distillation move
  the frontier substantially, but they do not remove model-capacity,
  data-coverage, or domain-transfer limits.
