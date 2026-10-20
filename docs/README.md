# Documentation index

- [Qwen3.8-27B monitor training](findings/qwen38_27b_monitor_training.md):
  trained 27B reaches 0.922 ID pAUROC@20 (+3.5 over unadapted); OOD pending.
- [First trained Qwen3.8-27B](findings/qwen38_27b_campaign.md): 0.9217 ID and
  0.9087 OOD pAUROC@20 (student prompt); 0.9352 under the teacher prompt.
- [Qwen3.8-27B training default](decisions/qwen38_27b_b200_training_default.md):
  BF16/SDPA with trim and 12 stored attention layers, 25.0 s per benchmark
  update (−10.1%).
- [Qwen 27B generations OOD](findings/qwen27b_generations_ood.md): unadapted
  Qwen3.8-27B leads 3.6 and 3.5 (0.890 / 0.873 / 0.839 Mean-OOD pAUROC@20).
- [Distillation temperature](findings/distillation_temperature.md): T=2/4 KD
  and target sharpening all lower ID pAUROC@20; keep T = 1.
- [Opt-in prefix-cached serving](decisions/prefix_cached_serving.md):
  synchronous-monitoring mode with accepted cached-score drift.
- [BF16/SDPA training default](decisions/b200_bf16_sdpa_training_default.md):
  replaces the FP4/FA4 default after the teacher precision factorial.
- [Precision selection on OOD](decisions/precision_selection_on_ood.md): precision
  and kernel choices may select on the frozen OOD suite under frozen rules.
- [QAT campaign shutdown](findings/qat_shutdown.md): verified B200 termination,
  retained NC2 volume and checksummed runtime/adapter preservation for resumption.
- [Full QAD OOD follow-up](findings/qad_ood.md): fixed full-epoch checkpoint,
  BF16 transfer improvement but a substantial optimized-serving penalty.
- [Filtered FP4/FA4 OOD evaluation](findings/filtered_fp4_ood.md): fixed original
  adapter, unchanged OOD inputs and two BF16-only passes against cached controls.
- [Network-volume checkpoint cleanup](findings/network_volume_cleanup.md):
  seven reconstructable merges pruned, preserving verified bases and FP32 adapters.
- [Short QAD budgets](findings/qad_budget.md): 9/17-minute recovery arms improve
  optimized ID ranking but retain less recovery than the full 66-minute run.
- [Direct Kimi QAT](findings/kimi_direct_qat.md): completed fresh-adapter
  distillation with worse ID ranking than direct and two-stage baselines.
- [Quantization-aware recovery](findings/quantization_recovery.md): same-model
  BF16 distillation, merged-weight deployment simulation and fresh numerical gates.
- [QAT training optimization](findings/qat_training_optimization.md): matched
  MLP/GDN contraction screens, rejected native FP8 parity and Kimi cache support.
- [Native FP8 QAT and MXFP8 backward](findings/native_fp8_qat.md): faster training
  with an ID quality regression, plus successful native-backward attention probes.
- [Native MXFP8 backward QAT](findings/native_mxfp8_qat.md): 6.1% faster full-model
  training with a further ID ranking regression against FA4 surrogate backward.

- Agent instructions by task: [compute and operations](agent_guides/compute.md),
  [training](agent_guides/training.md), and [inference/evaluation](agent_guides/inference.md).
  The root `AGENTS.md` defines when each guide must be read.
- [Research program](research_program.md): current direction, guiding conclusions,
  evaluation contract and open questions.
- [SHADE missing assistant prose](findings/shade_missing_prose.md): prepared
  inputs omit reports/commentary; raw content mixes prose and deliberation.
- [OOD prose restoration](findings/ood_prose_restoration.md): original AISI
  message boundaries, exact-rollout matching and remaining recovery gaps.
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
- [Qwen3.8-27B serving transfer](research/qwen38_27b_serving_transfer.md):
  backlog note on which 4B serving optimizations transfer to the 27B, what
  needs adapting or revalidating, and a four-step plan; not launched.
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
- [pplx-decider v1.2 OOD benchmark](findings/pplx_decider_ood_benchmark.md): local
  native-readout evaluation of the 27B and Flash on the Jev requests; the 27B is the
  lowest-cost frontier point.
- [Decision models OOD benchmark](findings/decision_models_ood_benchmark.md):
  Microsoft-Decision-1, Mercury Decide, d1-ultra and d1 on the Jev requests; none
  reaches the frontier.
- `infrastructure.md`: cluster, Lambda, secrets, caches, and operational commands.
- `decisions.md`: lightweight chronological decision log.
- `mats_project_log.md`: append-only session log for the 20-hour MATS
  application project, with cumulative personal time and commit evidence.
- `writeups/gleipnir_project_writeup.md`: canonical working draft of the
  project narrative before export to Google Docs.
- `writeups/gleipnir_lesswrong_writeup.md`: faithful copy of the submitted MATS
  project narrative for small clarity edits before web publication.
- [Mechanistic injection writeup](writeups/mechanistic_injection_writeup.md):
  curated summary of activation-based data filtering and CAFT, with
  supporting steering, transfer and precision controls.
- `findings/`: durable per-experiment findings; create one document per track.
- [Blackwell inference search](findings/blackwell_inference_search.md): frozen
  development vLLM/kernel search, frozen full-512 FP8 confirmation, independent
  engine restarts and rejected native FP4 layouts on one RTX PRO 6000 allocation.
- [Monitor serving precision](decisions/b200_monitor_score_reference.md):
  future-experiment inference/evaluation recipe, supported context and model/GPU
  envelope, parity requirements and frozen B200 comparison controls.
- [SDPA injection-direction replication review](research/sdpa_injection_direction_replication.md):
  colleague's clamp/projection results, recall tradeoffs, Lens requirements and
  user-authorized small APPS fit contract; the linked experiment freezes the
  current-model replication and disjoint task test partition.
- [SDPA injection-direction results](findings/b200_injection_direction.md):
  completed clamp/projection/random and additive controls, APPS false-alarm/
  recall tradeoffs, ID guard, grouped uncertainty and independent receipts.
- [Projection transfer to JudgeDeceiver](findings/b200_projection_judge.md):
  unchanged APPS direction in original A/B judging, small mixed accuracy gains,
  calibration regression and matched readout/identity checks; fixed-direction
  captures and equal-length benign controls show partial, heterogeneous signal
  transfer despite near-complete geometric removal; numeric-output follow-up
  shows prompt sensitivity and answer-0 bias without reliable projection benefit.
- [Base-model JudgeDeceiver steering](findings/base_judge_steering.md): the
  augmented-monitor vector strongly biases unadapted judging toward B while
  retaining its APPS alarm effect on the same base checkpoint.
- [Training-example direction alignment](findings/b200_training_direction.md):
  complete original-data base/SDPA census, source and teacher–label candidate
  concentrations, full rankings and review flags; earlier small augmented-pair
  diagnostic remains separate, with no causal attribution or applied filtering.
- [Firewall-content census](findings/training_firewall_census.md): full 9B logit
  screen, source/teacher-conditioned associations, classifier misses and
  inspected high-change counterexamples without firewall content.
- [Teacher-disagreement filtering](findings/teacher_disagreement_filtering.md):
  strict and 0.05-margin filtering improve one-seed ID ranking, with higher
  fixed-threshold false alarms; wider margins regress Gloom.
- [Strict-filtering OOD transfer](findings/teacher_disagreement_ood.md): small
  macro ranking gain over cached Gleipnir 4B, with source regressions and
  worse Brier; one-seed historical-serving comparison limits attribution.
- [Default strict teacher filtering](decisions/strict_teacher_filtering.md):
  selected training-data rule, resolved campaign preparation and control opt-out.
- [FP32 monitor-head resolution](findings/monitor_score_resolution.md): most
  strict-model score ties disappear, with small ID/OOD ranking gains and nearly
  unchanged measured throughput; no serving-default promotion.
- [Matched 4B/9B teacher fit](findings/monitor_training_loss_fit.md): final training
  KL is very similar, with a 4.2% pooled 9B advantage; no capacity ceiling established.
- [Single partial-AUROC variant](findings/partial_auroc.md): a small tail-ranking
  term improves OOD pAUROC but regresses ID Gloom; no default promotion.
- [Activation-ranked training removal](findings/activation_filter_training.md):
  top-20% harmless removal and matched random control improve ID but worsen
  held-out APPS injection discrimination; fixed-axis projection on the filtered
  checkpoint partly recovers ranking; model-specific refitting lowers calibrated
  FPR with substantial recall loss and a large calibration shift.
- [Concept-ablation training](findings/concept_ablation_training.md): regular-data
  training with the fixed injection axis removed at every residual layer, then
  ordinary inference; original-axis and refitted projections leave alarms intact,
  while adding the original vector strongly raises clean/neutral honest alarms.
- [vLLM Lens monitor integration](findings/b200_vllm_lens.md): request-scoped
  residual capture/steering, chunk and batch isolation, eager numerical baseline
  and matched overhead measurements.
- [Strict-filtered FP4/FA4 baseline](findings/filtered_fp4_baseline.md): restored
  NC2 runtime, completed BF16/optimized ID comparisons, compiler repair and
  serving speed/quality/repeatability tradeoffs against the existing SDPA control.
- [Augmented training precision controls](findings/b200_augmented_training.md): exact
  augmentation replay, completed FP32 adapters and failed optimized parity;
  ID/APPS tradeoffs, BF16 serving recovery, matched BF16 MLP/FA4 training and
  BF16/optimized ID/APPS and JudgeDeceiver comparisons, plus the historical
  adapter's current-stack BF16 ID control, authorized optimized ID diagnostic
  and bitwise-identical seed-0 BF16/FA4 training repeat; restoring SDPA partly
  recovers ID ranking without reproducing the historical checkpoint, while
  increasing APPS injection alarms and reducing matched discrimination.
- [Current monitor on APPS](findings/b200_apps.md): injected-honest alarms,
  matched backdoor discrimination and frozen optimized-stack evaluation.
- [Black-box APPS injection detection](findings/apps_injection_detection.md):
  matched base/SDPA BF16 presence prompting, near-perfect honest-code ranking,
  weak fixed-threshold recall and residual backdoor-related false positives.
- [Safety/injection score fusion](findings/apps_score_fusion.md): task-disjoint
  offline logistic combinations improve pooled behavior ranking but worsen
  matched injected-honest/backdoor discrimination; no serving promotion.
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
- [Short-request host overhead](findings/b200_host_overhead.md): C1 steps are
  host-bound; gated GDN route and lean GDN host path (both adopted) save ~5 ms;
  lean custom-op dispatch is within noise; remaining host time is diffuse.
- [Length-gated GDN prefill route](findings/b200_gdn_route.md): non-CP below 8K
  tokens per call saves 2.3–2.9 ms on short C1 requests (paired, same server);
  long chunks unchanged bitwise. Adopted in the FP8 default.
- [Kernel profile of the adopted default](findings/b200_kernel_profile.md):
  MXFP8 attention is 60–84% of GPU time from 256K; FP4 GEMMs and GDN dominate
  mid-length/c128; short requests are host-bound. Ranked candidates follow.
- [Last-query final-layer attention](findings/b200_last_layer_trim.md): adopted
  in the FP8 default; bitwise unchanged C1 scores, 6.4–10.3% lower C1 latency
  from 256K to 1M and 5.7% higher same-node c128 throughput. Finished-only
  selection adds nothing; the output-projection/MLP trim drifts with no gain.
- [B200 FP8 default contexts to 1M](findings/b200_context_million.md): C1
  1K–1M sweep with node/environment receipts, YaRN beyond 256K, chunk cost fit
  (attention 90% at 1M) and power-capped clocks on long attention.
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
- [Teacher filtering campaign shutdown](findings/teacher_sweep_shutdown.md):
  verified adapter/cache retention and confirmed NC2 B200 termination.
- `findings/monitoring_length_shortcut_audit.md`: source-specific length cues,
  paired inert-padding sensitivity, and the decision not to prioritize a
  length-specific regularizer.
- [SHADE-Arena length audit](findings/shade_length_audit.md): the upstream
  28K filter's 43% SHADE-Arena cut, and length-dependent OOD monitor quality
  after task/generator stratification.
- `migrations/`: historical material carried over from precursor projects.

Experiment READMEs describe executable work. Documents here record conclusions
that should influence later experiments.
