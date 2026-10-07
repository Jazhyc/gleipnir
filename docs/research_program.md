# Research program

## Goal

Build a broadly useful monitoring foundation model for AI control. It should
score deception, covert goal pursuit, sabotage, exfiltration, manipulation,
unsafe capability use and other strategically harmful behavior from model
interactions, while retaining uncertainty in ambiguous cases. Generalization
across tasks, model families and deployment settings is the central objective.
Qwen 3.5 is the initial backbone; the research goal is independent of that choice.

## Working approach

Distill strong teachers' judgments into smaller monitors using
parameter-efficient training and a direct binary decision-token interface.
Keep three supervision channels distinct: source hard labels, teacher soft
distributions, and privileged rationales or evidence. Matched comparisons are
needed to identify the value of each channel.

The completed tool-trajectory line uses Kimi K3 soft targets to train Qwen3.5
students on agent trajectories. Gleipnir 4B and 9B are released research artifacts;
see the [root README](../README.md) and
[experiment contract](../experiments/tool_trajectory_monitoring/README.md).
The intended monitoring surface includes visible messages, commands, tool calls,
outputs, edits and final reports. Historical inputs also contain identified
private reasoning, so action-only claims require the input audits and cleanup
contract below. Teachers may retain richer information, with that privilege
recorded explicitly in target provenance.

## Conclusions that guide new work

- **Data and ID improvements do not guarantee transfer.** The completed
  [soft-distillation scaling study](research/tool_trajectory_distillation_ood_scaling.md)
  has a non-monotonic one-seed OOD curve. More rows, larger models and stronger
  ID scores should be tested against grouped holdouts rather than assumed to
  improve generalization.
- **Injection text can trigger alarms without harmful behavior.**
  [Prompt changes](findings/monitoring_injection_disentanglement.md) and
  [aggressive training-data filtering](findings/monitoring_injection_removal.md)
  did not resolve injected-honest false positives in the tested APPS controls.
  Include honest injected negatives and matched harmful/harmless examples;
  suspicious wording alone is insufficient evidence of harmful behavior.
- **Direct monitoring augmentation is promising, with substantial limits.**
  [4B augmentation](findings/monitor_injection_augmentation.md) improves clean ID
  ranking and reduces injected-honest APPS alarms. The
  [9B replication](findings/monitor_injection_augmentation_9b.md) improves clean
  ID ranking but transfers much less strongly and regresses vanilla APPS
  ranking. These single-seed, limited-template, off-policy interventions do not
  establish broad or adaptive robustness, or a monotonic benefit from model size.
- **Quality judging and harmfulness monitoring have different targets.**
  [Joint preference training](findings/judge_injection_joint.md) learns the
  injected preference task without resolving APPS harmless alarms.
  [Auxiliary judge evaluation](findings/augmented_judge_evaluation.md) shows
  partial transfer from monitoring augmentation with substantial residual
  sensitivity. Keep preference accuracy separate from harmfulness diagnostics.
- **Systems gains require separate quality evidence.** Training and serving
  optimizations have explicit numerical acceptances and failed strict receipts.
  Small training-seen timing cohorts cannot establish held-out quality parity.
  The [full-training replication](decisions/b200_fp4_full_training_replication.md)
  improves ID ranking but worsens ECE; it is a one-seed combined-recipe comparison.

## Evaluation contract

Use the accepted [CoT-removed ID dataset](decisions/cot_removed_id_validation.md)
for future monitoring validation, hyperparameter comparisons, early stopping
and checkpoint selection. Remove source-typed reasoning and explicit thinking
calls while preserving visible assistant prose, other tool calls and results.
Freeze input/manifest hashes and keep distinct cache identities. This does not
retroactively clean training data or regenerate teacher targets. Historical
full-input scores and broader prose-removal ablations are different contracts.

- Group splits by available conversation, generator, task, source and annotation
  lineage; keep derived views with their parents and disclose missing lineage.
- Use source-macro raw normalized pAUROC@20 for the monitoring ID contract.
  Also report AUROC, calibration, threshold diagnostics, score ties and results
  by task/source/model family. Preserve the metric convention; standardized
  partial AUC and raw partial area divided by 0.2 are different quantities.
- Maintain honest hard negatives and ordinary-capability controls so factual
  mistakes are not conflated with deception or misalignment. Keep APPS out of
  training and checkpoint selection for these injection comparisons.
- Freeze selection and promotion criteria before evaluation. Keep the six-source
  OOD suite outside prompt, recipe, threshold, checkpoint and stopping choices;
  disclose prior use when interpreting later results. Never promote on final
  test results.
- Preserve raw teacher evidence, prompt hashes and target provenance. Distinguish
  uncertainty caused by missing observable evidence from presentation changes
  that truly preserve the label. Evaluate transfer before further scaling.

## Open research questions

1. Can training on the intended visible inputs, evidence-preserving consistency
   or compact grounded evidence supervision improve transfer with existing
   teacher data? The [training follow-up backlog](research/monitoring_training_followups.md)
   contains candidate designs, not a selected campaign.
2. Does broader injection wording and attack diversity reduce residual false
   positives without losing harmful-action recall? This follow-up remains
   deferred at the user's request; augmentation mechanisms are not yet isolated.
3. How do teacher quality, supervision channel, unique-data volume and repeated
   exposure affect ranking and calibration under matched comparisons?
4. Which domain mixtures improve general monitoring abstractions while retaining
   performance on earlier domains, and how should the ontology represent
   ambiguous evidence and deferral?

## Where details belong

Experiment READMEs hold hypotheses, executable contracts and run instructions.
[Findings and decisions](README.md) hold durable evidence and current selections;
research proposals live under `research/`. Follow the task-specific
[training](agent_guides/training.md), [inference](agent_guides/inference.md) and
[compute](agent_guides/compute.md) instructions for execution. Keep hardware
recipes, changing baselines, process state, timings and restart history there.

Update this program when evidence changes project direction or evaluation
principles. The [archived program](research_program_history.md) preserves the
previous chronology; future per-run updates belong in experiment records.
