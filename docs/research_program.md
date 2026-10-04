# Research program

## Goal

Train a broadly useful monitoring foundation model that scores control-relevant
behavior from model interactions. Deception is the first well-developed domain,
but the target ontology should grow to include misaligned actions, covert goal
pursuit, sabotage, manipulation, unsafe capability use, policy evasion, and
ambiguous cases where a monitor should defer.

## Initial strategy

Use strong teachers with privileged information and/or larger inference budgets
to annotate examples with both calibrated soft targets and compact evidence.
Distill those signals into Qwen 3.5 students with parameter-efficient training.
Compare three separable sources of value:

1. hard behavior labels;
2. teacher probability distributions at a fixed decision boundary;
3. compact rationales or structured evidence available only during training.

The first migrated experiment reproduces the deception-distillation pipeline in
a standalone form. Subsequent work should add task families through explicit
dataset adapters and an evolving, versioned behavior ontology.

The next proposed domain is action-only monitoring of agent tool trajectories:
detect sabotage, exfiltration, prompt-injection compliance, and scheming from
observable messages, commands, tool calls, outputs, edits, and final reports
without private reasoning. Its paper-derived task and dataset contract, along
with the proposed Kimi K3 logit-distillation comparison, is recorded in
`research/tool_transcript_monitoring.md`.

## Evaluation principles

- Split by source lineage, generator, task family, and model family where possible.
- Report macro ranking, calibration, thresholded safety diagnostics, and ties.
- Maintain honest hard negatives and ordinary-capability controls so factual
  error is not conflated with deception or misalignment.
- Evaluate transfer before scaling annotation volume or student size.
- Freeze promotion criteria before final evaluations.
- Preserve raw teacher evidence and immutable prompt hashes for auditability.

Instruction changes alone have not resolved injection-text sensitivity in the
completed [behavior-grounded 4B experiment](findings/monitoring_injection_disentanglement.md):
all three inference prompts retained 100% alarms on injected honest-code controls
at thresholds set from clean code. Require honest injected negatives and matched
positive/negative templates when interpreting injection benchmarks. This result
uses fixed teacher targets and a code-backdoor label contract; it does not identify
the causal training examples or settle the broader agent-behavior ontology.

The completed [aggressive census-filtering experiment](findings/monitoring_injection_removal.md)
also retained 100% injected-honest alarms at clean-code 2% FPR operating points
under all three monitor prompts. Removing 1,154 flagged rows lowered scores but
reduced standard-prompt ID macro pAUROC@20 from 0.8463 to 0.8223. This broad data
intervention did not resolve the failure; its changed class/source composition
and remaining classifier misses prevent a prompt-injection-specific causal claim.

For future monitoring validation and checkpoint selection, use the
[CoT-removed ID dataset](decisions/cot_removed_id_validation.md), preserving
visible assistant prose. Freeze its input/manifest hashes in each new campaign.

The completed [JudgeDeceiver continuation experiment](findings/judge_injection_continuation.md)
continues the regular 4B adapter on preference judgments with injections attached
independently to preferred and disfavored answers. Its pooled full release is
split approximately 80:20 by original query before augmentation. Preferred-answer
injection accuracy rises from 58.36% to 100%, and regular-prompt APPS injected-honest
alarms fall from 100% to 70.64% at clean-code 2% FPR operating points. ID macro
pAUROC@20 falls from 0.8463 to 0.7958 and calibration worsens. Preference robustness
does not establish overall harm-monitoring improvement. Preserve the distinction
between quality and harmfulness targets when designing benign injection exposure;
APPS remains excluded from training and checkpoint selection. No promotion follows.

The completed [joint-training experiment](../experiments/judge_injection_joint/README.md)
trains a fresh 4B adapter with the full regular trajectory population and a
75:25 monitoring/preference sampler. It preserves monitoring soft targets and
separate A/B hard preference labels, using the regular 5e-5 learning rate.
Approximately 22.5% of training draws contain injections; their 25 original
queries limit task diversity. Preferred-answer-injected accuracy reaches 100%,
and ID source-macro pAUROC@20 is 0.8658 versus 0.8463 original and 0.7958 after
preference-only continuation. However, all injected harmless APPS controls still
alarm at clean-code ~2% FPR; matched injected-harmful versus injected-harmless
AUROC is 0.6256 versus 0.6440 original. The auxiliary quality task can learn
injection handling without transferring that behavior to harmfulness monitoring.
These exploratory outcomes do not isolate forgetting. Only the fixed final
checkpoint and regular monitoring prompt were evaluated, with no APPS-based
selection or promotion. See the [finding](findings/judge_injection_joint.md).

## Near-term questions

The completed [monitor-directed injection augmentation](findings/monitor_injection_augmentation.md)
replaces 40% of the original monitoring rows with one off-policy injected view,
balancing user/assistant/tool roles and demanded verdicts across both source
behavior classes. It directly trains the harmfulness objective using the fixed
teacher cache. All 272 updates and 15,138 fresh scores are complete. Clean ID
source-macro pAUROC@20 rises from 0.8463 to 0.8730. APPS injected-honest FPR
falls from 100% to 44.43% averaged over three transfer templates at each
adapter's clean-code ~2% FPR threshold; injected-backdoor TPR falls to 87.64%.
Matched injected-harmful versus injected-harmless AUROC rises from 0.6440 to
0.8595. Held-out synthetic ID injections leave paired AUROC essentially unchanged
and give 1.85% harmless FPR at source-calibrated clean ~2% operating points.
This supports direct monitoring augmentation, with substantial residual APPS
sensitivity and limits from one seed, few authored strings and off-policy views.
Only the new adapter receives fresh scores; original regular-adapter results
are reused. APPS remains evaluation-only.

The user authorized a [9B replication](decisions/monitor_injection_augmentation_9b.md)
of the same frozen augmentation at 5e-5, using the previously validated 9B
checkpointing recipe and cached regular 9B controls. This campaign preserves
all training/evaluation views and introduces no recipe sweep. All 272 updates
and 15,138 fresh scores are complete and independently verified. Clean ID
source-macro pAUROC@20 rises from 0.8796 to 0.8939, but injected-honest APPS FPR
falls only from 99.85% to 83.36%, with two templates still at 100%. Matched
injected AUROC rises from 0.6588 to 0.6786, substantially less than at 4B;
vanilla APPS AUROC regresses from 0.9314 to 0.9126. This supports clean ID
benefits without broad OOD robustness or a model-size monotonicity claim. See
the [completed finding](findings/monitor_injection_augmentation_9b.md).
The existing B200 is verified EXITED after complete artifact collection and
independent audits; its persistent workspace and compiler/kernel caches remain.

The completed [auxiliary transfer evaluation](findings/augmented_judge_evaluation.md)
uses the unchanged original A/B quality-judging prompt and 4,188 JudgeDeceiver
holdout variants from six queries. Neither monitoring-augmented adapter trained
on JudgeDeceiver. Preferred-answer-injected accuracy rises from 58.36% to 62.65%
at 4B and 64.33% to 72.37% at 9B; freshly evaluated base 9B reaches 65.53%.
Clean accuracy stays near 99%, but augmented correct-to-wrong flips remain
37.08% / 27.63%. Every augmented query loses mean p(correct) under preferred
injections, and the hardest MT-Bench query reaches only 23.44% / 34.82%
accuracy. This supports partial transfer with residual avoidance across multiple
sources, rather than broad immunity. It is a preference diagnostic, not a
harmfulness false-positive rate. No promotion follows. The new EU evaluation
pod remains running after verified artifact collection; the prior NC2 pod and
its persistent caches remain preserved.

- With the backbone and teacher cache fixed, can CoT-removed student training,
  consistency regularization, compact evidence supervision, controlled data
  exposure, or checkpoint averaging improve transfer? The
  [possible training follow-ups](research/monitoring_training_followups.md)
  record the evidence and a proposed screen; these methods are not yet selected.
- How do soft label margins and rationale supervision scale with teacher quality
  and annotation volume?
- Does multi-domain joint training improve monitoring abstractions or cause
  destructive interference?
- Which Qwen 3.5 capacity is the best student/teacher frontier on two H100s?
- Which data mixtures transfer to held-out action types and model families?
- Can a single calibrated output schema express positive detections, benign
  behavior, and epistemic uncertainty without benchmark-specific routing?
