# Control Monitoring Has an Interface Problem

## Simple logit distillation for stronger, more general monitors

> Status: Working draft. This Markdown file is the canonical source for the
> project writeup until it is exported to Google Docs.

## 1. Background: training control monitors

### The monitoring problem

- Introduce control monitoring as the task of detecting scheming, sabotage,
  exfiltration, prompt-injection compliance, and other unauthorized behavior in
  an agent's observable trajectory.
- Explain the action-only setting: the monitor sees messages, tool calls,
  commands, outputs, edits, and the final report, but not private reasoning.
- Motivate the practical constraint: a useful monitor must identify harmful
  behavior while producing few false alarms.

### Sinha et al.'s setup

- Introduce Sinha et al., *Training Deliberative Monitors for Black-Box
  Scheming Detection*, as the main experimental reference.
- Describe their full monitoring specification and 0--10 severity interface.
  The monitor writes a visible rationale and then produces an ordinal score.
- Summarize their training pipeline: generate multiple teacher rationales,
  judge and filter them, supervise students on the selected rationales, and
  optionally continue training with reinforcement learning.
- Summarize the evaluation suite: 6,395 examples from six held-out sources
  covering different forms of agent misbehavior.
- Explain why this interface is understandable in an API-dominated field.
  Frontier APIs usually expose generated text but do not reliably expose the
  complete logits needed for direct decision-boundary scoring.
- Present this as context rather than a criticism of the paper: the generated
  rationale and ordinal score form a portable interface across closed models.

### The evaluation metric

- Define AUROC as a ranking metric: it measures how often positive examples
  receive higher scores than negative examples across possible thresholds.
- Define pAUROC@20 as the normalized area under the ROC curve up to a false
  positive rate of 20%.
- Explain why the low-false-positive region matters for monitoring: a monitor
  that generates too many false alarms is difficult to use for oversight.
- State the paper's primary aggregate: compute pAUROC@20 separately for each of
  the six OOD datasets and take their unweighted mean.
- Note that the comparison also tracks uncached inference cost per 1,000
  evaluations.

### Our starting point: Aletheia's Quest

- Make clear that Gleipnir did not start from a blank slate.
- State that the core hypotheses and training infrastructure were developed
  during the Aletheia's Quest deception-monitoring competition, where our
  methods placed first on the public leaderboard.
- Summarize the inherited recipe: teacher decision-token probabilities,
  QLoRA, direct soft-target training, and evaluation through continuous
  decision scores.
- Frame Gleipnir as a rapid transfer test: port the recipe to action-only
  control monitoring, preserve the paper's OOD evaluation contract, and test
  the highest-value hypotheses first.

### Why build a strong black-box baseline?

- State the project goal directly: build the strongest practical monitoring
  baseline available from observable model behavior alone.
- Explain why this matters for interpretability research. Novel methods can be
  scientifically interesting without improving operational monitoring, and
  weak baselines make it difficult to tell where an apparent gain comes from.
- Use Sparse Autoencoders as a motivating example: their representations may
  be useful for understanding models, but novelty and interpretability do not
  by themselves establish an advantage on control-relevant detection tasks.
- Be explicit that black-box output monitoring is not a complete solution to
  misalignment. A model can hide relevant intent or computation while
  producing behavior that looks benign.
- Explain the promise of white-box monitoring: access to activations and other
  internal signals may reveal information that is absent from observable
  outputs.
- Define the comparison standard for future work: a white-box method should
  beat a strong black-box monitor under matched data, evaluation, and compute
  before its gain is attributed to privileged internal information.
- Position Gleipnir as that control. Making the black-box baseline stronger
  raises the bar for future interpretability methods and makes their unique
  contribution easier to identify.

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
