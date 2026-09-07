# Documentation index

- `research_program.md`: scope, research questions, and evaluation principles.
- [Possible monitoring training follow-ups](research/monitoring_training_followups.md):
  fixed-backbone, existing-teacher-data proposals, their empirical and literature
  motivation, and a candidate comparison plan; research backlog rather than a
  launched campaign or selected recipe.
- `research/tool_transcript_monitoring.md`: operational paper notes, dataset
  provenance, and the proposed exploratory Kimi-logit scaling design for
  action-only agent monitoring.
- `research/tool_trajectory_inference_economics.md`: paper-compatible marginal
  inference-cost accounting for Qwen3.5 and the completed Kimi K3 and K2.6
  API baselines, plus a future prefill-only serving optimization backlog.
- `infrastructure.md`: cluster, Lambda, secrets, caches, and operational commands.
- `decisions.md`: lightweight chronological decision log.
- `mats_project_log.md`: append-only session log for the 20-hour MATS
  application project, with cumulative personal time and commit evidence.
- `writeups/gleipnir_project_writeup.md`: canonical working draft of the
  project narrative before export to Google Docs.
- `writeups/gleipnir_lesswrong_writeup.md`: faithful copy of the submitted MATS
  project narrative for small clarity edits before web publication.
- `findings/`: durable per-experiment findings; create one document per track.
- `findings/competition_poster_evidence.md`: frozen deception-monitor evidence
  tables and reporting qualifications for the competition poster.
- `findings/lambda_shutdown_inventory.md`: checksummed local artifact inventory
  retained before releasing the former two-H100 `monitor-foundation` instance.
- `findings/gleipnir_control_shutdown_inventory.md`: checksummed local artifact
  inventory retained before releasing the single-H100 `gleipnir-control`
  instance.
- [Improvement campaign inventory](findings/gleipnir_improvement_shutdown_inventory.md):
  pre-shutdown artifact checks, final ablation outcomes, and work that can resume
  later without retaining the two-H100 `gleipnir-improvement` instance.
- `findings/monitoring_length_shortcut_audit.md`: source-specific length cues,
  paired inert-padding sensitivity, and the decision not to prioritize a
  length-specific regularizer.
- `migrations/`: historical material carried over from precursor projects.

Experiment READMEs describe executable work. Documents here record conclusions
that should influence later experiments.
