# Documentation index

- `research_program.md`: scope, research questions, and evaluation principles.
- [Monitoring sequence packing](research/monitoring_sequence_packing.md):
  prior Phoenix rejection, model/kernel boundary audit, CPU isolation controls,
  and the B200 BF16 integration and correctness gates.
- [Packed BF16 B200 default](decisions/b200_packed_bf16_training_recipe.md):
  the selected packed training recipe without model checkpointing.
- [BF16 sequence packing](findings/bf16_sequence_packing.md): native sequence
  isolation, numerical drift localization, and the bounded B200 packing comparison.
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
- [GPT-6 Luna OOD benchmark](findings/openai_luna_ood_benchmark.md): full teacher
  prompt, Standard API, complete 6,395-row audit, cost, and bounded API recoveries.
- [GPT-5.6 Luna OOD benchmark](findings/openai_56_luna_ood_benchmark.md): matched
  full teacher prompt, 80 workers, complete audit, ranking and calibration tradeoffs.
- [Jev 1.13 OOD benchmark](findings/tool_trajectory_jev_ood_benchmark.md): retained
  native probability evaluation, complete artifact audit, exact cost and context caveat.
- `infrastructure.md`: cluster, Lambda, secrets, caches, and operational commands.
- `decisions.md`: lightweight chronological decision log.
- `mats_project_log.md`: append-only session log for the 20-hour MATS
  application project, with cumulative personal time and commit evidence.
- `writeups/gleipnir_project_writeup.md`: canonical working draft of the
  project narrative before export to Google Docs.
- `writeups/gleipnir_lesswrong_writeup.md`: faithful copy of the submitted MATS
  project narrative for small clarity edits before web publication.
- `findings/`: durable per-experiment findings; create one document per track.
- [Blackwell inference search](findings/blackwell_inference_search.md): frozen
  development vLLM/kernel search, frozen full-512 FP8 confirmation, independent
  engine restarts and rejected native FP4 layouts on one RTX PRO 6000 allocation.
- [B200 Four Over Six training handoff](findings/b200_fouroversix_training.md#resume-checklist-and-stability-candidates):
  completed native kernel checks, two failed full-model compilation gates,
  artifact/code state, and untested per-token scaling, dequantized BF16 backward
  and final-layer precision proposals for resuming the deferred investigation.
- `findings/competition_poster_evidence.md`: frozen deception-monitor evidence
  tables and reporting qualifications for the competition poster.
- `findings/lambda_shutdown_inventory.md`: checksummed local artifact inventory
  retained before releasing the former two-H100 `monitor-foundation` instance.
- `findings/gleipnir_control_shutdown_inventory.md`: checksummed local artifact
  inventory retained before releasing the single-H100 `gleipnir-control`
  instance.
- [Improvement campaign inventory](findings/gleipnir_improvement_shutdown_inventory.md):
  verified artifact retention, final ablation outcomes, and confirmed termination
  of the two-H100 `gleipnir-improvement` instance.
- `findings/monitoring_length_shortcut_audit.md`: source-specific length cues,
  paired inert-padding sensitivity, and the decision not to prioritize a
  length-specific regularizer.
- `migrations/`: historical material carried over from precursor projects.

Experiment READMEs describe executable work. Documents here record conclusions
that should influence later experiments.
