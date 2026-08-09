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

## Evaluation principles

- Split by source lineage, generator, task family, and model family where possible.
- Report macro ranking, calibration, thresholded safety diagnostics, and ties.
- Maintain honest hard negatives and ordinary-capability controls so factual
  error is not conflated with deception or misalignment.
- Evaluate transfer before scaling annotation volume or student size.
- Freeze promotion criteria before final evaluations.
- Preserve raw teacher evidence and immutable prompt hashes for auditability.

## Near-term questions

- How do soft label margins and rationale supervision scale with teacher quality
  and annotation volume?
- Does multi-domain joint training improve monitoring abstractions or cause
  destructive interference?
- Which Qwen 3.5 capacity is the best student/teacher frontier on two H100s?
- Which data mixtures transfer to held-out action types and model families?
- Can a single calibrated output schema express positive detections, benign
  behavior, and epistemic uncertainty without benchmark-specific routing?

