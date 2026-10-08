# Documentation index

- Agent instructions by task: [compute and operations](agent_guides/compute.md),
  [training](agent_guides/training.md), and [inference/evaluation](agent_guides/inference.md).
  The root `AGENTS.md` defines when each guide must be read.
- [Research program](research_program.md): current direction, guiding conclusions,
  evaluation contract and open questions.
- [Archived research program](research_program_history.md): historical chronology
  retained before the 2026-10-07 simplification; includes superseded status notes.
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
- [Optimized vLLM serving default](decisions/b200_monitor_score_reference.md):
  future-experiment inference/evaluation recipe, supported context and model/GPU
  envelope, parity requirements and frozen B200 comparison controls.
- [vLLM Lens monitor integration](findings/b200_vllm_lens.md): request-scoped
  residual capture/steering, chunk and batch isolation, eager numerical baseline
  and matched overhead measurements.
- [Augmented training precision controls](findings/b200_augmented_training.md): exact
  augmentation replay, completed FP32 adapters and failed optimized parity;
  ID/APPS tradeoffs, BF16 serving recovery, matched BF16 MLP/FA4 training and
  same-adapter BF16/optimized ID/APPS and JudgeDeceiver serving comparisons.
- [Current monitor on APPS](findings/b200_apps.md): injected-honest alarms,
  matched backdoor discrimination and frozen optimized-stack evaluation.
- [vLLM 0.31 migration](findings/b200_vllm031.md): source-compatible B200 runtime,
  failed score agreement, paired native arithmetic, scheduler diagnostics and
  fixed-adapter ID ranking/calibration comparison.
- [Full-attention precision recovery](findings/b200_attention_precision.md):
  frozen FP4/BF16/FP8 projection comparison, partial ID pAUROC recovery and
  development/ID quality, calibration and throughput tradeoffs.
- [FP8 GDN projections](findings/b200_gdn_precision.md): small ID ranking gain,
  improved calibration, development low-FPR regression and batch throughput cost.
- [Direct FP4 SwiGLU-output ablation](findings/b200_swiglu_output.md): restored
  whole-row scaling improves development ranking but worsens full ID pAUROC.
- [Optimized-stack ID drift](findings/b200_optimized_id.md): paired same-adapter
  ID regression, recovery groups and continuous-admission throughput/latency tradeoff.
- [B200 length-aware admission](findings/b200_length_admission.md): modest
  low-load short-request benefit, longer long-request tails and failed initial
  latency/fairness/pooled-AUROC screen on the selected scorer.
- [B200 reference concurrency scaling](findings/b200_score_scaling.md): same
  full320 workload at c1/2/4/8/16/32/64/128, throughput plateau near c16, latency
  growth and batch-dependent score diagnostics; current FP8/0.31 NC2 host
  comparison, c32 plateau and matched-recipe c128 timing/quality control.
- [B200 fixed 2K context scaling](findings/b200_context_scaling.md): cropped
  exact-2048-token systems workload, higher request throughput, plateau near
  c16/32 and unchanged warm reference.
- [B200 single-request long contexts](findings/b200_long_context.md): exact
  8K–256K capacity/latency sweep, bounded temporary memory, and an opt-in fix for
  the pinned pooling scheduler's exact-cap stall.
- [B200 BF16 GDN state](findings/b200_gdn_state.md): native state admission,
  gate-rounding diagnostic and negligible full-vLLM throughput change.
- [B200 dedicated monitor scoring](findings/b200_monitor_score_endpoint.md):
  two-logit cached classification endpoint, measured output costs and the
  negative speed result with batch-quality qualifications.
- [B200 large-prefill graphs](findings/b200_prefill_graph_capture.md):
  NC2 and repaired EU retries, replay profiles and negative batch speed results.
- [B200 Triton mutation analysis](findings/b200_triton_mutation_analysis.md):
  source-bound Torch/Triton metadata repair, exact native parity and warning-free
  serving measurements, with failed probes and attribution limits.
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
