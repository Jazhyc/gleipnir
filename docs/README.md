# Documentation index

Experiment READMEs describe executable work. Documents here record conclusions
that should influence later experiments. Every finding, decision and plan
starts with a `Status:` and `Date:` line; entries below are newest first.
Findings record completed measurements; decisions say whether they are
`current`, `superseded` or `historical`.

## Start here

- [Current state](current.md): what is selected, released, bound and open now.
- [Research program](research_program.md): goals, scientific direction and the
  evaluation contract ([history](research_program_history.md)).
- Agent instructions by task: [compute and operations](agent_guides/compute.md),
  [training](agent_guides/training.md) and [inference](agent_guides/inference.md).
- [Infrastructure](infrastructure/README.md): workstation, cluster, Lambda,
  Runpod and OpenRouter procedures.
- [Decisions](decisions/README.md): every decision, dated.
- [Experiments index](../experiments/README.md).

## Monitor training

Decisions:

- [Train Gleipnir 4B on tool-trajectory data only](decisions/training/train_gleipnir_4b_on_tool_trajectory_data_only.md) (2026-10-20): User decision: Gleipnir 4B uses the same training data as Gleipnir 27B.
- [Advance `2e-5` for monitoring-only LR confirmation](decisions/training/advance_2e_5_for_monitoring_only_lr_confirmation.md) (2026-09-04) *(historical)*: The five-cell seed-0 Qwen3.5-4B screen selected AdamW `2e-5` over the freshly retrained `5e-5` control on the frozen 3,012-row ID development suite.
- [Run tool-trajectory scaling as Kimi-soft-only distillation](decisions/training/run_tool_trajectory_scaling_as_kimi_soft_only_distillation.md) (2026-08-30) *(historical)*: Train only Kimi K3 soft-logit students for the first Qwen3.5-9B tool-trajectory scaling campaign.
- [Do not promote hard-only from the internal split](decisions/training/do_not_promote_hard_only_from_the_internal_split.md) (2026-08-17) *(historical)*: The unadapted Qwen3.5-9B base reaches 0.92273 macro AUROC on the exact 1,972-row internal development surface (0.88903 pooled AUROC).
- [Screen trajectory, regularization, and data weighting](decisions/training/screen_trajectory_regularization_and_data_weighting.md) (2026-08-14) *(historical)*: After Muon failed full-data confirmation, keep AdamW and screen sixteen rank-128 cells: one shared baseline plus fifteen single-factor changes to warmup, schedule, epoch …
- [Screen structural training interventions](decisions/training/screen_structural_training_interventions.md) (2026-08-14) *(historical)*: After conventional optimizer and regularization changes failed to improve macro AUROC, test twenty rank-128 cells on the fixed internal development protocol: three …
- [Retain the baseline after the regularization screen](decisions/training/retain_the_baseline_after_the_regularization_screen.md) (2026-08-14) *(historical)*: All sixteen rank-128 screen cells completed on the fixed internal development split.
- [Retain AdamW after full-data Muon confirmation](decisions/training/retain_adamw_after_full_data_muon_confirmation.md) (2026-08-13) *(historical)*: Neither predeclared Muon candidate passed the frozen three-seed confirmation gate.
- [Confirm the two Muon candidates on full data](decisions/training/confirm_the_two_muon_candidates_on_full_data.md) (2026-08-13) *(historical)*: Replace the planned 50%-data confirmation with a 100%-training-data confirmation at the user's direction.
- [Compare Muon through scheduled update-RMS matching](decisions/training/compare_muon_through_scheduled_update_rms_matching.md) (2026-08-13) *(historical)*: Replace the independent, unscheduled Muon learning rate with the standard scheduler-managed parameter-group `lr`.
- [Advance two Muon rates to replicated confirmation](decisions/training/advance_two_muon_rates_to_replicated_confirmation.md) (2026-08-13) *(historical)*: The rank-64, 25%-data internal-development screen completed all ten paired cells.
- [Measure data and adapter capacity jointly](decisions/training/measure_data_and_adapter_capacity_jointly.md) (2026-08-12) *(historical)*: Cross nested `10%`, `25%`, and `50%` training selections with QLoRA ranks 4, 16, 64, and 256 using paired seeds 0, 1, and 2.
- [Train text-only LoRAs causally and export serving copies](decisions/training/train_text_only_loras_causally_and_export_serving_copies.md) (2026-08-11): For experiments containing no image or video inputs, train Qwen3.5 LoRAs through its causal text compatibility model and preserve that FP32 adapter as the master …
- [Require FLA and standard QLoRA for the rank curve](decisions/training/require_fla_and_standard_qlora_for_the_rank_curve.md) (2026-08-11) *(historical)*: The fast Qwen3.5 H100 path requires more than `AutoModelForCausalLM`: install isolated `flash-linear-attention==0.5.2` and `fla-core==0.5.2` packages before model …
- [Match multimodal training and serving architectures](decisions/training/match_multimodal_training_and_serving_architectures.md) (2026-08-10) *(superseded)*: Load multimodal backbones such as Qwen3.5 through `AutoModelForImageTextToText` during training, even for text-only batches, and restrict LoRA targets explicitly to the …

Findings:

- [First trained Qwen3.8-27B monitor](findings/training/qwen38_27b_campaign.md) (2026-10-18): 0.9217 ID and 0.9087 OOD pAUROC@20 (student prompt); 0.9352 under the teacher prompt.
- [Qwen3.8-27B monitor training (first campaign)](findings/training/qwen38_27b_monitor_training.md) (2026-10-17): soft 27B 0.922 ID / 0.909 OOD; matched hard labels 0.967 ID / 0.904 OOD; in the release configuration soft wins OOD 0.951 vs 0.933.
- [Distillation temperature](findings/training/distillation_temperature.md) (2026-10-16): T=2/4 KD and target sharpening all lower ID pAUROC@20; keep T = 1.
- [One strict-filtered partial-AUROC variant](findings/training/partial_auroc.md) (2026-10-12): a small tail-ranking term improves OOD pAUROC but regresses ID Gloom; no default promotion.
- [Matched 4B/9B teacher fit](findings/training/monitor_training_loss_fit.md) (2026-10-12): final training KL is very similar, with a 4.2% pooled 9B advantage; no capacity ceiling established.
- [Tool-trajectory hard-label strength screen](findings/training/monitoring_hard_labels.md) (2026-10-03): 2026-10-03 (Amsterdam time).
- [Objective accumulation audit](findings/training/monitoring_objective_accumulation.md) (2026-09-05): both corrected rationale reruns and their ID evaluations completed.
- [Monitoring-only Qwen3.5-4B learning-rate screen](findings/training/monitoring_lr_sweep.md) (2026-09-04): The learning rate transferred from the earlier mixed-data recipe is not optimal for one-epoch, rank-128 QLoRA distillation on only the 8,688 monitoring trajectories.
- [Qwen3.5-9B held-out ID data-mixture comparison](findings/training/gleipnir9b_id_data_mixture.md) (2026-09-04): Adding the 13,149-example deception corpus to the same 8,688 monitoring examples improves the existing seed-0 Qwen3.5-9B adapter on the frozen 3,012-row held-out ID …
- [Monitoring-only 4B LR sweep preflight](findings/training/monitoring_lr_sweep_preflight.md) (2026-09-03): Before launching the learning-rate screen, the v1 recipe tested rank-128 QLoRA with microbatch 8 and accumulation 4 on the 32 longest monitoring examples.
- [Gleipnir 4B held-out ID baseline](findings/training/gleipnir4b_held_out_id_baseline.md) (2026-09-03): Before selecting new training hyperparameters, evaluate the completed mixed-data Gleipnir 4B adapter and its unadapted Qwen3.5-4B backbone on the exact held-out …
- [Qwen3.5-0.8B mixed-distillation systems preflight](findings/training/tool_trajectory_qwen08b_stopped_systems_preflight.md) (2026-08-31): The planned Qwen3.5-0.8B mixed-data soft-distillation and OOD campaign was intentionally stopped on 2026-08-31.
- [Gleipnir 4B full-prompt OOD ablation](findings/training/tool_trajectory_gleipnir4b_full_prompt_ood_ablation.md) (2026-08-31): The mixed-data Gleipnir 4B adapter was trained on the compact student prompt against Kimi K3 distributions produced with the full teacher monitoring specification.
- [Soft-distillation OOD data scaling](findings/training/tool_trajectory_distillation_ood_scaling.md) (2026-08-31): This experiment measures strict-OOD monitoring as matched tool-trajectory training volume increases under the inherited deception recipe.
- [Tool-trajectory auxiliary findings](findings/training/tool_trajectory_auxiliary_findings.md) (2026-08-31): This document collects exploratory and supporting results that inform the research program but are not primary evaluation endpoints.

## Teachers and training data

Decisions:

- [Strict teacher agreement is the default training-data filter](decisions/teachers/strict_teacher_filtering.md) (2026-10-11): selected training-data rule, resolved campaign preparation and control opt-out.
- [Make paid teacher caching provider-sticky and auditable](decisions/teachers/make_paid_teacher_caching_provider_sticky_and_auditable.md) (2026-08-30): Route synchronous OpenRouter annotation campaigns through a stable `session_id` derived from the model and exact prompt prefix, mark that prefix with an explicit …

Findings:

- [Strict-filtered FP4/FA4 OOD evaluation](findings/teachers/filtered_fp4_ood.md) (2026-10-13): fixed original adapter, unchanged OOD inputs and two BF16-only passes against cached controls.
- [Filtered FP4/FA4 baseline campaign](findings/teachers/filtered_fp4_baseline.md) (2026-10-12): restored NC2 runtime, completed BF16/optimized ID comparisons, compiler repair and serving speed/quality/repeatability tradeoffs against the existing SDPA control.
- [Strict teacher-disagreement filtering: OOD transfer](findings/teachers/teacher_disagreement_ood.md) (2026-10-11): small macro ranking gain over cached Gleipnir 4B, with source regressions and worse Brier; one-seed historical-serving comparison limits attribution.
- [Teacher-disagreement filtering on monitoring training data](findings/teachers/teacher_disagreement_filtering.md) (2026-10-11): strict and 0.05-margin filtering improve one-seed ID ranking, with higher fixed-threshold false alarms; wider margins regress Gloom.
- [Qwen3.8-Flash matched direct-logprob comparison](findings/teachers/qwen38flash_matched_teacher_comparison.md) (2026-09-07): 2026-09-07.
- [MiniMax M3 matched full-boundary comparison](findings/teachers/minimax_matched_teacher_comparison.md) (2026-09-07): 2026-09-07.
- [Kimi K2.6 matched training-population diagnostic](findings/teachers/kimi26_matched_teacher_comparison.md) (2026-09-07): 2026-09-07.
- [Matched Qwen 3.5 27B and Kimi K3 teacher agreement](findings/teachers/matched_qwen_kimi_teacher_agreement.md) (2026-09-06): 2026-09-06.
- [Kimi K2.6 Inceptron OOD binary-logit benchmark](findings/teachers/tool_trajectory_kimi_k26_ood_benchmark.md) (2026-09-01): Unadapted `moonshotai/kimi-k2.6`, served exclusively by Inceptron's dated `moonshotai/kimi-k2.6-20260420` endpoint, reaches **0.9194 Mean-OOD pAUROC@20** and **0.9751 …
- [Kimi K3 training-mixture teacher cache](findings/teachers/tool_trajectory_training_teacher_cache.md) (2026-08-30): in-distribution diagnostic on labeled training data, not a held-out benchmark.
- [Kimi K3 teacher-cache cost estimates](findings/teachers/tool_trajectory_teacher_cost_estimate.md) (2026-08-30): paid launch.
- [Kimi K3 binary-logit teacher canary](findings/teachers/tool_trajectory_teacher_canary.md) (2026-08-30): an OOD evaluation or a replication of the paper's final data manifest.
- [Kimi K3 scalar structured-output canary](findings/teachers/tool_trajectory_structured_output_canary.md) (2026-08-30): strict structured condition did **not** meet the frozen logit-parity rule.
- [Kimi K3 OOD binary-logit canary](findings/teachers/tool_trajectory_kimi_ood_canary.md) (2026-08-30): evaluation authorized and ready to scale.
- [Kimi K3 OOD binary-logit benchmark](findings/teachers/tool_trajectory_kimi_ood_benchmark.md) (2026-08-30): This benchmark measures `moonshotai/kimi-k3` through the Fireworks provider on the six paper-author OOD action-only datasets.

## Evaluation and base-model screens

Decisions:

- [Inkling 2 Mini as the next large-monitor base](decisions/evaluation/inkling2_mini_next_base.md) (2026-11-23): user selection 2026-11-23; unadapted 0.884 OOD vs Qwen3.8-27B 0.890, faster than the optimized 27B; vLLM batching and training support still open.
- [Plot Gleipnir at optimized-serving performance; no QAD for the 27B](decisions/evaluation/plot_gleipnir_at_optimized_serving_performance_no_qad_for.md) (2026-10-20): User decision: the frontier registry enters Gleipnir models at their performance on the project's optimized serving stack, because project serving prices will set their …
- [Evaluate Gleipnir 27B under the full teacher prompt](decisions/evaluation/evaluate_gleipnir_27b_under_the_full_teacher_prompt.md) (2026-10-20): User decision: by default, Gleipnir 27B is evaluated under the full teacher prompt, not the compact student prompt it was trained on.
- [Precision and kernel choices may be selected on the OOD suite](decisions/evaluation/precision_selection_on_ood.md) (2026-10-15): precision and kernel choices may select on the frozen OOD suite under frozen rules.
- [Retain the frozen OOD inputs for paper comparability](decisions/evaluation/frozen_ood_comparability.md) (2026-10-12): User selection, 2026-10-12: keep the existing six-source, 6,395-row OOD inputs and reported scores unchanged so comparisons retain the paper-inspired input contract.
- [Use CoT-removed ID inputs for future validation](decisions/evaluation/cot_removed_id_validation.md) (2026-09-08): Decision: accepted by the user on 2026-09-08.
- [Default frozen text evaluation to vLLM](decisions/evaluation/default_frozen_text_evaluation_to_vllm.md) (2026-08-17): Use a persistent vLLM engine with continuous batching for large frozen text-only base/LoRA evaluations.

Findings:

- [SleightBench: released monitors, single pass](findings/evaluation/sleightbench_released_monitors.md) (2026-10-21): single-pass 4B/27B scores on hand-built evasion attacks at 262K context; 15% (27B) and 5% (4B) catch at 1% FPR against 32% for Opus 4.6 with thinking; training data lacks monitor-adversarial, mistake-shaped attacks.
- [Hosted decision models on the strict-OOD suite](findings/evaluation/decision_models_ood_benchmark.md) (2026-10-20): Microsoft-Decision-1, Mercury Decide, d1-ultra and d1 on the Jev requests; none reaches the frontier.
- [pplx-decider v1.2 native-readout OOD benchmark](findings/evaluation/pplx_decider_ood_benchmark.md) (2026-10-19): local native-readout evaluation of the 27B and Flash on the Jev requests; the 27B is the lowest-cost frontier point.
- [SHADE-Arena length audit](findings/evaluation/shade_length_audit.md) (2026-10-18): the upstream 28K filter's 43% SHADE-Arena cut, and length-dependent OOD monitor quality after task/generator stratification.
- [Qwen 27B generations on the strict OOD suite](findings/evaluation/qwen27b_generations_ood.md) (2026-10-17): unadapted Qwen3.8-27B leads 3.6 and 3.5 (0.890 / 0.873 / 0.839 Mean-OOD pAUROC@20).
- [SHADE assistant-prose omission audit](findings/evaluation/shade_missing_prose.md) (2026-10-12): prepared inputs omit reports/commentary; raw content mixes prose and deliberation.
- [Recovering the original OOD public monitoring surface](findings/evaluation/ood_prose_restoration.md) (2026-10-12): original AISI message boundaries, exact-rollout matching and remaining recovery gaps.
- [FP32 final-head computation on the strict-filtered monitor](findings/evaluation/monitor_score_resolution.md) (2026-10-11): most strict-model score ties disappear, with small ID/OOD ranking gains and nearly unchanged measured throughput; no serving-default promotion.
- [Tool-trajectory monitor inference economics](findings/evaluation/tool_trajectory_inference_economics.md) (2026-09-30): paper-compatible marginal inference-cost accounting for Qwen3.5 and the completed Kimi K3 and K2.6 API baselines, plus a future prefill-only serving optimization backlog.
- [GPT-6 Luna full-teacher Standard API OOD benchmark](findings/evaluation/openai_luna_ood_benchmark.md) (2026-09-30): full teacher prompt, Standard API, complete 6,395-row audit, cost, and bounded API recoveries.
- [Direct OpenAI GPT-6 Luna logprob compatibility canary](findings/evaluation/openai_luna_logprobs_canary.md) (2026-09-30): Follow-up corrected the initial missing-score diagnosis: explicit `top_p=1` restored both label scores on all three Responses examples.
- [GPT-5.6 Luna full-teacher Standard API OOD benchmark](findings/evaluation/openai_56_luna_ood_benchmark.md) (2026-09-30): matched full teacher prompt, 80 workers, complete audit, ranking and calibration tradeoffs.
- [Qwen3.5-35B-A3B mixed-provider full-teacher-prompt OOD benchmark](findings/evaluation/tool_trajectory_qwen35b_a3b_ood_benchmark.md) (2026-09-19): The unadapted `Qwen/Qwen3.5-35B-A3B` tokenizer/model route uses tokenizer revision `59d61f3ce65a6d9863b86d2e96597125219dc754`, OpenRouter model ID …
- [Jev 1.13 native probability OOD benchmark](findings/evaluation/tool_trajectory_jev_ood_benchmark.md) (2026-09-19): retained native probability evaluation, complete artifact audit, exact cost and context caveat.
- [Frozen OOD input reasoning audit](findings/evaluation/ood_input_reasoning.md) (2026-09-08): no changes to data, prompts, labels, evaluation results, or models.
- [Reasoning in monitoring teacher and student training inputs](findings/evaluation/training_input_reasoning.md) (2026-09-07): the 13,149 additional prior-deception examples in the standard mixed checkpoint are outside this audit.
- [Pretrained-only Qwen3.5 ID evaluation](findings/evaluation/pretrained_qwen_id.md) (2026-09-07): Completed and audited 2026-09-07.
- [ID assistant prose and reasoning audit](findings/evaluation/id_assistant_prose_and_reasoning.md) (2026-09-07): The frozen 3,012-row ID monitor inputs contain private reasoning.
- [Monitoring length-shortcut audit](findings/evaluation/monitoring_length_shortcut_audit.md) (2026-09-04): source-specific length cues, paired inert-padding sensitivity, and the decision not to prioritize a length-specific regularizer.
- [Qwen3.5-9B visible-reasoning ordinal OOD benchmark](findings/evaluation/tool_trajectory_qwen_ordinal_reasoning_ood_benchmark.md) (2026-08-31): On the frozen six-source, 6,395-row strict OOD suite, unadapted `Qwen/Qwen3.5-9B` reached **0.6656 Mean-OOD pAUROC@20** and **0.8610 Mean-OOD AUROC** when it visibly …
- [Qwen3.5-9B visible-reasoning binary-logit OOD benchmark](findings/evaluation/tool_trajectory_qwen_binary_reasoning_ood_benchmark.md) (2026-08-31): On the frozen six-source, 6,395-row strict OOD suite, unadapted `Qwen/Qwen3.5-9B` reached **0.6923 Mean-OOD pAUROC@20** and **0.8607 Mean-OOD AUROC** after one visible …
- [Qwen3.5-27B full-teacher-prompt OOD benchmark](findings/evaluation/tool_trajectory_qwen27b_ood_benchmark.md) (2026-08-31): This benchmark measures the unadapted `Qwen/Qwen3.5-27B@fc05daec18b0a78c049392ed2e771dde82bdf654` model on the six paper-author OOD action-only datasets.
- [GPT-OSS-120B low-reasoning OOD benchmark](findings/evaluation/tool_trajectory_gpt_oss_low_reasoning_ood_benchmark.md) (2026-08-31): This condition uses the same pinned `openai/gpt-oss-120b` revision, complete teacher prompt, and six-source OOD artifact as the immediate-logit baseline.
- [Qwen3.5 reasoning on held-out in-distribution trajectories](findings/evaluation/tool_trajectory_qwen_reasoning_id.md) (2026-08-30): The native-thinking arm was cancelled before evaluation because of its projected inference cost.
- [Qwen3.5 reasoning-mode tool-trajectory canary](findings/evaluation/tool_trajectory_qwen_reasoning_canary.md) (2026-08-30): not the paper's final STRIDE test or Gloom-Exfiltration evaluation.
- [Qwen3.5-9B immediate ordinal-score OOD benchmark](findings/evaluation/tool_trajectory_qwen_ordinal_ood_benchmark.md) (2026-08-30): On the frozen six-source, 6,395-row strict OOD suite, unadapted `Qwen/Qwen3.5-9B` reached **0.4858 Mean-OOD pAUROC@20** and **0.7592 Mean-OOD AUROC** when it generated …
- [Qwen3.5-9B full-teacher-prompt OOD benchmark](findings/evaluation/tool_trajectory_qwen_ood_benchmark.md) (2026-08-30): This benchmark measures the unadapted `Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a` base on the six paper-author OOD action-only datasets.
- [Qwen3.5-4B full-teacher-prompt OOD benchmark](findings/evaluation/tool_trajectory_qwen4b_ood_benchmark.md) (2026-08-30): This benchmark measures the unadapted `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` model on the six paper-author OOD action-only datasets.
- [Tool-trajectory OOD cost--performance frontier](findings/evaluation/tool_trajectory_ood_frontier.md) (2026-08-30): This file is the canonical comparison table for completed evaluations on the six-source, 6,395-row strict OOD suite from Sinha et al., *Training Deliberative Monitors …
- [GPT-OSS-120B full-teacher-prompt OOD benchmark](findings/evaluation/tool_trajectory_gpt_oss_ood_benchmark.md) (2026-08-30): This benchmark measures the unadapted `openai/gpt-oss-120b@b5c939de8f754692c1647ca79fbf85e8c1e70f8a` base on the six paper-author OOD action-only datasets.
- [Competition poster evidence](findings/evaluation/competition_poster_evidence.md) (2026-08-18): frozen deception-monitor evidence tables and reporting qualifications for the competition poster.

## Adversarial stress tests

Findings:

- [Gleipnir against frontier black-box monitors](findings/adversarial/adversarial_frontier_comparison.md) (2026-10-23): level with Kimi K3, Sol and Sonnet on natural cheats at 3–13× lower cost; below chance under misleading code comments, where Sol and Sonnet hold.
- [Can a cheap monitor be turned against the agents it watches?](findings/adversarial/adversarial_lab_v03.md) (2026-10-20): Gleipnir 4B in a six-agent lab · Abhinav Kumar · 20 October 2026*

## Prompt-injection robustness

Decisions:

- [Replicate monitor-directed injection augmentation on 9B](decisions/injection/monitor_injection_augmentation_9b.md) (2026-10-04) *(historical)*: all required transfers to the existing Runpod B200, and shutdown after training, evaluation and verified collection.
- [Monitor-directed injection replacement](decisions/injection/monitor_injection_augmentation.md) (2026-10-04) *(historical)*: Recorded: user selected replacement; CPU preparation completed.

Findings:

- [Released monitors on APPS monitor-directed injections](findings/injection/apps_released_monitors.md) (2026-10-21): both releases flag every injected honest program at the APPS operating point; the adapter improves on the base 27B; projecting out an injection direction only softens the 4B's response.
- [Combining behavior and injection-presence scores](findings/injection/apps_score_fusion.md) (2026-10-11): task-disjoint offline logistic combinations improve pooled behavior ranking but worsen matched injected-honest/backdoor discrimination; no serving promotion.
- [Black-box APPS injection detection](findings/injection/apps_injection_detection.md) (2026-10-11): matched base/SDPA BF16 presence prompting, near-perfect honest-code ranking, weak fixed-threshold recall and residual backdoor-related false positives.
- [Concept-ablation training on regular monitoring data](findings/injection/concept_ablation_training.md) (2026-10-10): regular-data training with the fixed injection axis removed at every residual layer, then ordinary inference; original-axis and refitted projections leave alarms intact, while adding the original vector strongly raises clean/neutral honest alarms.
- [Augmented-vector steering in base-model JudgeDeceiver judging](findings/injection/base_judge_steering.md) (2026-10-10): the augmented-monitor vector strongly biases unadapted judging toward B while retaining its APPS alarm effect on the same base checkpoint.
- [Activation-ranked training removal](findings/injection/activation_filter_training.md) (2026-10-10): top-20% harmless removal and matched random control improve ID but worsen held-out APPS injection discrimination; fixed-axis projection on the filtered checkpoint partly recovers ranking; model-specific refitting lowers calibrated FPR with substantial recall loss and a large calibration shift.
- [Firewall content and learned direction alignment](findings/injection/training_firewall_census.md) (2026-10-09): full 9B logit screen, source/teacher-conditioned associations, classifier misses and inspected high-change counterexamples without firewall content.
- [Training-example alignment with the APPS direction](findings/injection/b200_training_direction.md) (2026-10-09): complete original-data base/SDPA census, source and teacher–label candidate concentrations, full rankings and review flags; earlier small augmented-pair diagnostic remains separate, with no causal attribution or applied filtering.
- [APPS projection transfer to JudgeDeceiver](findings/injection/b200_projection_judge.md) (2026-10-09): unchanged APPS direction in original A/B judging, small mixed accuracy gains, calibration regression and matched readout/identity checks; fixed-direction captures and equal-length benign controls show partial, heterogeneous signal transfer despite near-complete geometric removal; numeric-output follow-up shows prompt sensitivity and answer-0 bias without reliable projection benefit.
- [SDPA injection-direction intervention replication](findings/injection/b200_injection_direction.md) (2026-10-09): completed clamp/projection/random and additive controls, APPS false-alarm/ recall tradeoffs, ID guard, grouped uncertainty and independent receipts.
- [vLLM Lens on the pooling monitor](findings/injection/b200_vllm_lens.md) (2026-10-08): request-scoped residual capture/steering, chunk and batch isolation, eager numerical baseline and matched overhead measurements.
- [Augmented monitor training precision controls](findings/injection/b200_augmented_training.md) (2026-10-08): exact augmentation replay, completed FP32 adapters and failed optimized parity; ID/APPS tradeoffs, BF16 serving recovery, matched BF16 MLP/FA4 training and BF16/optimized ID/APPS and JudgeDeceiver comparisons, plus the historical adapter's current-stack BF16 ID control, authorized optimized ID diagnostic and bitwise-identical seed-0 BF16/FA4 training repeat; restoring SDPA partly recovers ID ranking without reproducing the historical checkpoint, while increasing APPS injection alarms and reducing matched discrimination.
- [Current FP4-trained monitor on APPS](findings/injection/b200_apps.md) (2026-10-08): injected-honest alarms, matched backdoor discrimination and frozen optimized-stack evaluation.
- [Monitor-directed injection augmentation at 9B](findings/injection/monitor_injection_augmentation_9b.md) (2026-10-04): and independently verified.
- [Monitor-directed injection replacement campaign](findings/injection/monitor_injection_augmentation.md) (2026-10-04): evaluation scores finished; the serving engine exited and the B200 is idle.
- [Kimi K3 on injected honest APPS code: collaborator-reported diagnostic](findings/injection/kimi_teacher_injection_controls.md) (2026-10-04): Recorded 2026-10-04 from Michael Kenney’s writeup supplied by Jeremias Ferrao.
- [Joint monitoring and injected-preference training](findings/injection/judge_injection_joint.md) (2026-10-04): User selected a fresh adapter with 75% monitoring and 25% preference examples, using the regular 5e-5 learning rate.
- [Monitoring augmentation transfers partially to judge injection handling](findings/injection/augmented_judge_evaluation.md) (2026-10-04): augmented judge evaluation.
- [Prompt injections in the materialized training traces](findings/injection/training_trace_prompt_injections.md) (2026-10-03): Reviewed 2026-10-03.
- [Full-training prompt-injection presence census](findings/injection/training_injection_census.md) (2026-10-03): Run 2026-10-03 with the user-authorized Runpod B200.
- [Census-filtered standard-prompt 4B: filtering did not resolve injection alarms](findings/injection/monitoring_injection_removal.md) (2026-10-03): 2026-10-03.
- [Behavior-grounded 4B monitor: prompt change did not remove injection sensitivity](findings/injection/monitoring_injection_disentanglement.md) (2026-10-03): 2026-10-03.
- [Judge-targeted preference continuation](findings/injection/judge_injection_continuation.md) (2026-10-03): `experiments/injection/judge_injection_continuation/README.md`.
- [Student prompt-injection awareness campaign](findings/injection/student_injection_awareness.md) (2026-10-02): The prespecified experiment compares regular and explicitly injection-aware student instructions at 4B and 9B, holding all 8,688 tool trajectories and cached Kimi K3 …
- [Crossed monitor prompt-injection evaluation](findings/injection/monitor_injection_evaluation.md) (2026-10-02): This review uses dataset descriptions and generation procedures, not a search through individual trajectories.

## Activation probes

Findings:

- [SleightBench: NTP activation probes, single pass](findings/probes/sleightbench_ntp_probes.md) (2026-10-26): frozen NTP activation probes, stripped view, single pass; AUROC 0.61 (hard) and 0.63 (soft) against 0.66 (release prompt) and 0.61 (ControlArena) for the 4B, catch 5–10%, more pair wins; long context explains little.

## Quantization-aware training

Decisions:

- [Selected merged-weight QAT training recipe](decisions/quantization/qat_reference_training.md) (2026-10-13): The user selects the original QAT arithmetic with redundant operations removed on 2026-10-13.

Findings:

- [Full QAD checkpoint OOD follow-up](findings/quantization/qad_ood.md) (2026-10-13): fixed full-epoch checkpoint, BF16 transfer improvement but a substantial optimized-serving penalty.
- [Short QAD budgets](findings/quantization/qad_budget.md) (2026-10-13): 9/17-minute recovery arms improve optimized ID ranking but retain less recovery than the full 66-minute run.
- [Direct Kimi K3 QAT](findings/quantization/kimi_direct_qat.md) (2026-10-13): completed fresh-adapter distillation with worse ID ranking than direct and two-stage baselines.
- [Quantization-aware recovery of the filtered 4B monitor](findings/quantization/quantization_recovery.md) (2026-10-12): same-model BF16 distillation, merged-weight deployment simulation and fresh numerical gates.
- [Merged QAT training systems optimization](findings/quantization/qat_training_optimization.md) (2026-10-12): matched MLP/GDN contraction screens, rejected native FP8 parity and Kimi cache support.
- [Native MXFP8 backward in full QAT](findings/quantization/native_mxfp8_qat.md) (2026-10-12): 6.1% faster full-model training with a further ID ranking regression against FA4 surrogate backward.
- [Native FP8 QAT and MXFP8 attention backward](findings/quantization/native_fp8_qat.md) (2026-10-12): faster training with an ID quality regression, plus successful native-backward attention probes.

## Training systems

Decisions:

- [Qwen3.8-27B B200 training default](decisions/systems/qwen38_27b_b200_training_default.md) (2026-10-17): BF16/SDPA with trim and 12 stored attention layers, 25.0 s per benchmark update (−10.1%).
- [Native FP4 MLP B200 training default](decisions/systems/b200_native_fp4_training_recipe.md) (2026-10-16) *(superseded)*: Superseded as the default on 2026-10-16 by BF16/SDPA; the profile below remains available as `qwen35_4b_b200_fp4_mlp`.
- [BF16/SDPA B200 training default](decisions/systems/b200_bf16_sdpa_training_default.md) (2026-10-16): replaces the FP4/FA4 default after the teacher precision factorial.
- [Latest-only weights for short systems training checks](decisions/systems/systems_training_adapter_retention.md) (2026-10-06): Decision date: 2026-10-06.
- [Select native FP4 MLPs as the B200 training default](decisions/systems/select_native_fp4_mlps_as_the_b200_training_default.md) (2026-10-05) *(superseded)*: `qwen35_4b_b200_default` now selects the warmed native NVFP4 MLP recipe with BF16 FlashQLA/FA4 and FP32 master adapters.
- [Resident B200 FP4 optimization baseline](decisions/systems/b200_fp4_optimization_worker.md) (2026-10-05) *(superseded)*: On 2026-10-05 the user selected combined native FP4 MLPs with BF16 FA4 as the baseline for further training-stack timing improvements and requested keeping a training …
- [Full regular 4B replication with native FP4 MLPs](decisions/systems/b200_fp4_full_training_replication.md) (2026-10-05) *(historical)*: Decision date: 2026-10-05.
- [Packed BF16 LoRA as the B200 training default](decisions/systems/b200_packed_bf16_training_recipe.md) (2026-10-04) *(superseded)*: the selected packed training recipe without model checkpointing.
- [Persistent compiler caches across training campaigns](decisions/systems/persistent_training_compiler_cache.md) (2026-10-03): User-selected standing preference, 2026-10-03: reuse warm persistent compiler and kernel caches across compatible training runs.
- [Select uniform FlashQLA for B200 NF4/BF16 LoRA](decisions/systems/select_uniform_flashqla_for_b200_nf4_bf16_lora.md) (2026-10-02) *(superseded)*: Use `systems_screen@_global_: qwen35_4b_b200_default`, with all 24 GDN layers on FlashQLA, SDPA full attention and the selected checkpoint/adaptive batching recipe.
- [FlashQLA for the default B200 LoRA recipe](decisions/systems/b200_flashqla_training_recipe.md) (2026-10-02) *(superseded)*: Recorded: selected by explicit user instruction, 2026-10-02.
- [Adaptive batching for future B200 training](decisions/systems/b200_adaptive_training_recipe.md) (2026-10-01) *(superseded)*: Recorded: adaptive batching selected by explicit user instruction, 2026-10-01.
- [Config-driven monitoring systems screens](decisions/systems/config_driven_systems_screens.md) (2026-09-04): Recorded: adopted 2026-09-04; Hydra authoring added 2026-09-04.
- [Enable pinned causal-conv1d for long Qwen3.5-4B training](decisions/systems/enable_pinned_causal_conv1d_for_long_qwen3_5_4b_training.md) (2026-09-03) *(historical)*: On the Stack 24 two-H100 `gleipnir-improvement` target, source-build pinned `causal-conv1d==1.6.2.post1` into an isolated target and require a BF16 GPU forward/backward …

Findings:

- [Direct FROST bindings in FP4 LoRA training](findings/systems/b200_frost_training.md) (2026-10-07): 2026-10-07.
- [Direct variable-length NVIDIA MXFP8 attention](findings/systems/b200_nvidia_mxfp8_varlen.md) (2026-10-05): 2026-10-05.
- [NVIDIA MXFP8 causal attention on B200](findings/systems/b200_nvidia_mxfp8.md) (2026-10-05): Strict native numerical parity fails.
- [Fused MXFP8 preparation and square block scaling](findings/systems/b200_mxfp8_fused.md) (2026-10-05): 2026-10-05.
- [B200 MLP / GEMM screen](findings/systems/b200_mlp_gemm.md) (2026-10-05): for large MLP / GEMM optimization, with FP4 as the eventual target.
- [Meta attention optimization audit and causal B200 screen](findings/systems/b200_meta_stack.md) (2026-10-05): 2026-10-05.
- [Eight-bit attention training kernel survey](findings/systems/b200_eight_bit_training_kernel_survey.md) (2026-10-05): No GPU execution, dependency changes, or new training runs were performed.
- [MXFP8 FlashAttention 4 integration feasibility](findings/systems/b200_mxfp8_fa4_feasibility.md) (2026-10-04): PyTorch article and Meta's public `lp_fa4` implementation at revision `2889aefba0d03b3eecf779eed9f4097066ecdc5d`.
- [Native FP4 MLPs after leaving QLoRA](findings/systems/b200_fp4_mlp_lora.md) (2026-10-04): work with the current packed LoRA recipe, but measured updates take 53.44% more time than the matched BF16 control (7.86192 versus 5.12363 seconds).
- [FlashAttention 4 screen on the packed BF16 B200 recipe](findings/systems/b200_bf16_fa4.md) (2026-10-04): 20.24% faster with FA4 (5.12363 to 4.08648 seconds); partitions and token counts match exactly.
- [Boundary-safe Qwen3.5 sequence packing](findings/systems/monitoring_sequence_packing.md) (2026-10-02): prior Phoenix rejection, model/kernel boundary audit, CPU isolation controls, and the B200 BF16 integration and correctness gates.
- [BF16 LoRA sequence packing on B200](findings/systems/bf16_sequence_packing.md) (2026-10-02): native sequence isolation, numerical drift localization, and the bounded B200 packing comparison.
- [FP4 training stability on the restored B200](findings/systems/fp4_training_stability.md) (2026-10-01): `fp4_stability`.
- [B200 training throughput and attention backend screen](findings/systems/b200_training_throughput.md) (2026-10-01): Subsequent decision (2026-10-01): the user selected compiled adaptive batching for future training after a separate matched execution audit.
- [Native Four Over Six MLP training on B200](findings/systems/b200_fouroversix_training.md) (2026-10-01): completed native kernel checks, two failed full-model compilation gates, artifact/code state, and untested per-token scaling, dequantized BF16 backward and final-layer precision proposals for resuming the deferred investigation.
- [B200 gradients, actual updates and short learning curves](findings/systems/b200_execution_audit.md) (2026-10-01): Subsequent decision (2026-10-01): after reviewing these results, the user explicitly selected compiled adaptive batching as the recipe for future B200 training.
- [Adaptive B200 physical microbatches](findings/systems/b200_adaptive_microbatching.md) (2026-10-01): Subsequent decision (2026-10-01): the user explicitly selected compiled 16,384-token/max-8 adaptive batching for future training.
- [Two-GPU mixed-corpus MIL throughput](findings/systems/monitoring_mil_ddp.md) (2026-09-07): On the two reserved H100 SXM5 80GB GPUs, eager two-process DDP with FLA and causal-conv1d outperformed the compiled single-GPU MIL recipe on a matched 256-parent mixture.
- [Qwen3.5 causal-conv1d on Lambda Stack 24](findings/systems/qwen35_causal_conv1d_stack24.md) (2026-09-03): Can the Qwen3.5 training stack use `causal-conv1d` reliably on the long-lived Lambda Stack 24 H100 target, and is the isolated kernel materially faster than …

## Inference and serving

Decisions:

- [Price Gleipnir release models at self-hosted B200 break-even](decisions/serving/price_gleipnir_release_models_at_self_hosted_b200_break_even.md) (2026-10-20): User decision: Gleipnir 4B v0.3 and Gleipnir 27B are priced at self-hosting break-even, one B200 at `$7.99/h` at 100% utilization.
- [Opt-in prefix-cached serving](decisions/serving/prefix_cached_serving.md) (2026-10-16): synchronous-monitoring mode with accepted cached-score drift.
- [Gate the GDN context-parallel prefill route by length](decisions/serving/gate_the_gdn_context_parallel_prefill_route_by_length.md) (2026-10-15): The FP8 default takes FlashInfer's non-CP GDN prefill below 8,192 tokens per call.
- [Adopt the lean GDN host path with a strided gated norm](decisions/serving/adopt_the_lean_gdn_host_path_with_a_strided_gated_norm.md) (2026-10-15): The FP8 default runs the lean GDN prefill host path with unchanged kernels.
- [Adopt last-query final-layer attention in the FP8 default](decisions/serving/adopt_last_query_final_layer_attention_in_the_fp8_default.md) (2026-10-15): The optimized serving default runs layer-31 attention only from each prefill request's last token.
- [Default future experiments to optimized vLLM](decisions/serving/default_future_experiments_to_optimized_vllm.md) (2026-10-08) *(superseded)*: Use the optimized serving implementation for future experiment inference, serving and evaluation.
- [Monitor serving precision and frozen comparisons](decisions/serving/b200_monitor_score_reference.md) (2026-10-08): future-experiment inference/evaluation recipe, supported context and model/GPU envelope, parity requirements and frozen B200 comparison controls.
- [Direct FP4-output B200 inference reference](decisions/serving/b200_native_fp4_output_inference_baseline.md) (2026-10-07) *(superseded)*: Selected by the user on 2026-10-07.
- [Native Gigatoken and direct FROST serving reference](decisions/serving/b200_gigatoken_direct_host_reference.md) (2026-10-07) *(superseded)*: Historical selection, superseded by the repaired score reference.
- [Combined preparation and tuned FP4 GEMMs as the B200 inference reference](decisions/serving/b200_tuned_fp4_inference_baseline.md) (2026-10-06) *(superseded)*: Subsequently superseded by the user-selected FP4 attention projection reference.
- [MXFP8 prefill as the B200 inference optimization reference](decisions/serving/b200_mxfp8_inference_baseline.md) (2026-10-06) *(superseded)*: Superseded later on 2026-10-06 by the user-selected combined-preparation/shape-tuned reference.
- [B200 serving precision after FROST baseline selection](decisions/serving/b200_low_precision_attention_serving.md) (2026-10-06) *(historical)*: baseline, prioritizing higher-concurrency throughput, and asks about lower precision for GDN/linear attention and full attention.
- [Native inference kernel screening on B200](decisions/serving/b200_inference_kernels.md) (2026-10-06) *(historical)*: Decision date: 2026-10-06.
- [Production inference development benchmark](decisions/serving/b200_inference_benchmark.md) (2026-10-06) *(historical)*: Decision date: 2026-10-06.
- [Reuse training FROST forward arithmetic in vLLM](decisions/serving/b200_frost_inference.md) (2026-10-06) *(historical)*: Decision date: 2026-10-06.
- [Native cuDNN FP4 MLP inference screen](decisions/serving/b200_fp4_inference.md) (2026-10-06) *(historical)*: Decision date: 2026-10-06.
- [FP4 GDN projections as the B200 serving baseline](decisions/serving/b200_fp4_gdn_inference_baseline.md) (2026-10-06) *(superseded)*: Superseded for new optimization comparisons on 2026-10-06 by the MXFP8 prefill reference.
- [FP4 attention projections as the B200 inference reference](decisions/serving/b200_attention_fp4_inference_baseline.md) (2026-10-06) *(superseded)*: Decision date: 2026-10-06.

Findings:

- [Native Gigatoken serving frontend](findings/serving/b200_gigatoken_frontend.md) (2026-11-15): Correction, 2026-11-15.** This integration replaces `BaseRenderer._encode`, which exists in vLLM 0.24 and not in vLLM 0.31.
- [Last-query attention in the final layer](findings/serving/b200_last_layer_trim.md) (2026-10-15): adopted in the FP8 default; bitwise unchanged C1 scores, 6.4–10.3% lower C1 latency from 256K to 1M and 5.7% higher same-node c128 throughput. Finished-only selection adds nothing; the output-projection/MLP trim drifts with no gain.
- [Kernel profile of the adopted last-query default](findings/serving/b200_kernel_profile.md) (2026-10-15): MXFP8 attention is 60–84% of GPU time from 256K; FP4 GEMMs and GDN dominate mid-length/c128; short requests are host-bound. Ranked candidates follow.
- [Short-request host overhead](findings/serving/b200_host_overhead.md) (2026-10-15): C1 steps are host-bound; gated GDN route and lean GDN host path (both adopted) save ~5 ms; lean custom-op dispatch is within noise; remaining host time is diffuse.
- [Length-gated GDN prefill route](findings/serving/b200_gdn_route.md) (2026-10-15): non-CP below 8K tokens per call saves 2.3–2.9 ms on short C1 requests (paired, same server); long chunks unchanged bitwise. Adopted in the FP8 default.
- [B200 FP8 default single-request contexts to 1M](findings/serving/b200_context_million.md) (2026-10-15): C1 1K–1M sweep with node/environment receipts, YaRN beyond 256K, chunk cost fit (attention 90% at 1M) and power-capped clocks on long attention.
- [vLLM 0.31 migration and scheduling diagnostics](findings/serving/b200_vllm031.md) (2026-10-08): source-compatible B200 runtime, failed score agreement, paired native arithmetic, scheduler diagnostics and fixed-adapter ID ranking/calibration comparison.
- [Direct FP4 SwiGLU-output ablation on the FP8-attention default](findings/serving/b200_swiglu_output.md) (2026-10-08): restored whole-row scaling improves development ranking but worsens full ID pAUROC.
- [Optimized-stack drift on the last full-trained adapter](findings/serving/b200_optimized_id.md) (2026-10-08): paired same-adapter ID regression, recovery groups and continuous-admission throughput/latency tradeoff.
- [FP8 GDN projections on the FP8-attention default](findings/serving/b200_gdn_precision.md) (2026-10-08): small ID ranking gain, improved calibration, development low-FPR regression and batch throughput cost.
- [Full-attention projection precision recovery](findings/serving/b200_attention_precision.md) (2026-10-08): frozen FP4/BF16/FP8 projection comparison, partial ID pAUROC recovery and development/ID quality, calibration and throughput tradeoffs.
- [Qwen Triton mutation-analysis repair](findings/serving/b200_triton_mutation_analysis.md) (2026-10-07): source-bound Torch/Triton metadata repair, exact native parity and warning-free serving measurements, with failed probes and attribution limits.
- [Tokenization contribution to B200 serving](findings/serving/b200_serving_tokenization.md) (2026-10-07): 2026-10-07.
- [B200 serving startup](findings/serving/b200_serving_startup.md) (2026-10-07): 2026-10-07.
- [B200 serving CPU placement](findings/serving/b200_serving_cpu_placement.md) (2026-10-07): 2026-10-07.
- [B200 selected reference concurrency scaling](findings/serving/b200_score_scaling.md) (2026-10-07): same full320 workload at c1/2/4/8/16/32/64/128, throughput plateau near c16, latency growth and batch-dependent score diagnostics; current FP8/0.31 NC2 host comparison, c32 plateau and matched-recipe c128 timing/quality control.
- [B200 large-prefill graph capture](findings/serving/b200_prefill_graph_capture.md) (2026-10-07): NC2 and repaired EU retries, replay profiles and negative batch speed results.
- [Prefill launch overhead and CUDA-graph coverage](findings/serving/b200_prefill_dispatch.md) (2026-10-07): Investigated 2026-10-07 on the saved Direct FP4 reference trace and archived installed vLLM 0.24.0 source.
- [Direct FP4-output serving bottlenecks](findings/serving/b200_native_fp4_output_profile.md) (2026-10-07): Measured 2026-10-07 on the retained NC2 B200, API/engine 95619/95657, port 8010.
- [Dedicated two-logit monitoring endpoint](findings/serving/b200_monitor_score_endpoint.md) (2026-10-07): two-logit cached classification endpoint, measured output costs and the negative speed result with batch-quality qualifications.
- [B200 single-request long contexts](findings/serving/b200_long_context.md) (2026-10-07): exact 8K–256K capacity/latency sweep, bounded temporary memory, and an opt-in fix for the pinned pooling scheduler's exact-cap stall.
- [B200 length-aware admission](findings/serving/b200_length_admission.md) (2026-10-07): modest low-load short-request benefit, longer long-request tails and failed initial latency/fairness/pooled-AUROC screen on the selected scorer.
- [BF16 GDN state in vLLM](findings/serving/b200_gdn_state.md) (2026-10-07): native state admission, gate-rounding diagnostic and negligible full-vLLM throughput change.
- [Direct GDN prefill output on the Direct FP4 reference](findings/serving/b200_gdn_direct_output.md) (2026-10-07): in the reference profile.
- [B200 FROST host binding wrappers](findings/serving/b200_frost_host_wrappers.md) (2026-10-07): 2026-10-07.
- [B200 fixed 2K context scaling](findings/serving/b200_context_scaling.md) (2026-10-07): cropped exact-2048-token systems workload, higher request throughput, plateau near c16/32 and unchanged warm reference.
- [B200 cache policy and restart comparison](findings/serving/b200_cache_policy_same_host.md) (2026-10-07): 2026-10-07.
- [Cache-free complete-prompt monitoring on B200](findings/serving/b200_cache_free_serving.md) (2026-10-07): cache-free prototype removes persistent cache storage.
- [SwiGLU overhead and direct FP4 output follow-up](findings/serving/b200_swiglu_overhead_native_output.md) (2026-10-06): from the existing fused producer, then try native FP4 output.
- [Forward-only cuDNN MXFP8 attention on B200](findings/serving/b200_mxfp8_serving.md) (2026-10-06): frozen inference screen.
- [Native FP4 gate/up GEMM and SwiGLU fusion](findings/serving/b200_fp4_swiglu_fusion.md) (2026-10-06): comparison.
- [FP4 activation preparation on the MXFP8 serving reference](findings/serving/b200_fp4_serving_preparation.md) (2026-10-06): installed CUDA per-token NVFP4 packing, fused SwiGLU packing, and fused post-attention residual/RMSNorm packing.
- [Shape-dependent FP4 GEMM tiles on B200](findings/serving/b200_fp4_gemm_tuning.md) (2026-10-06): for output projections, but no clear whole-serving win.
- [Matched FP4 GEMM backend comparison on B200](findings/serving/b200_fp4_backend_comparison.md) (2026-10-06): selecting the combined-preparation, shape-tuned FROST serving reference.
- [FP4 full-attention projections on B200](findings/serving/b200_attention_projection_fp4.md) (2026-10-06): projection GEMMs to FP4.
- [Runpod B200: released Gleipnir 4B ID verification](findings/serving/runpod_b200_gleipnir4b_id.md) (2026-09-30): `runpod_gleipnir4b_id`.
- [RTX PRO 6000 inference search — 2026-09-29](findings/serving/blackwell_inference_search.md) (2026-09-29): frozen development vLLM/kernel search, frozen full-512 FP8 confirmation, independent engine restarts and rejected native FP4 layouts on one RTX PRO 6000 allocation.

## Operations: provisioning, shutdowns, cleanup

Decisions:

- [Bootstrap](decisions/operations/bootstrap.md) (2026-08-09) *(historical)*: Start Gleipnir as a standalone repository rather than retaining competition submission and leaderboard machinery.

Findings:

- [QAT and OOD campaign shutdown](findings/operations/qat_shutdown.md) (2026-10-13): verified B200 termination, retained NC2 volume and checksummed runtime/adapter preservation for resumption.
- [Network-volume checkpoint cleanup](findings/operations/network_volume_cleanup.md) (2026-10-13): seven reconstructable merges pruned, preserving verified bases and FP32 adapters.
- [Teacher filtering campaign shutdown](findings/operations/teacher_sweep_shutdown.md) (2026-10-12): verified adapter/cache retention and confirmed NC2 B200 termination.
- [B200 provisioned for prospective quantization-aware training](findings/operations/b200_qat_provisioning.md) (2026-10-12): At the user's request, one B200 was provisioned on 2026-10-12 as Pod `tv6i08lhsf2cfy` (`gleipnir-b200-qat`) in US-CA-2 at $7.99/hour.
- [`gleipnir-improvement` shutdown inventory](findings/operations/gleipnir_improvement_shutdown_inventory.md) (2026-09-08): verified artifact retention, final ablation outcomes, and confirmed termination of the two-H100 `gleipnir-improvement` instance.
- [`gleipnir-control` shutdown artifact inventory](findings/operations/gleipnir_control_shutdown_inventory.md) (2026-09-01): checksummed local artifact inventory retained before releasing the single-H100 `gleipnir-control` instance.
- [Lambda shutdown artifact inventory](findings/operations/lambda_shutdown_inventory.md) (2026-08-18): checksummed local artifact inventory retained before releasing the former two-H100 `monitor-foundation` instance.

## Plans and designs

- [Adversarial stress tests of the released monitors](plans/adversarial_stress_testing.md) (2026-10-20): threat model, low-FPR metrics and the phase-1 attack plan for the released monitors.
- [Adversarial stress tests: run queue](plans/adversarial_run_queue.md) (2026-10-20): One list of what is running, what's next and what's parked.
- [Phase 2 design: multi-agent failure modes of per-agent monitors](plans/adversarial_phase2_design.md) (2026-10-20): Draft, 20 October 2026.
- [Does the monitor respond to what the code does, or to how the comments describe it?](plans/adversarial_code_or_comments.md) (2026-10-20): Design draft, 20 October 2026.
- [Transferring the 4B serving optimizations to Qwen3.8-27B](plans/qwen38_27b_serving_transfer.md) (2026-10-18): backlog note on which 4B serving optimizations transfer to the 27B, what needs adapting or revalidating, and a four-step plan; not launched.
- [SDPA monitor replication of injection-direction interventions](plans/sdpa_injection_direction_replication.md) (2026-10-09): colleague's clamp/projection results, recall tradeoffs, Lens requirements and user-authorized small APPS fit contract; the linked experiment freezes the current-model replication and disjoint task test partition.
- [Monitor injection evaluation: crossed model and student instruction](plans/monitor_injection_evaluation.md) (2026-10-02): Evaluate the completed matched tool-trajectory-only students on monitor-injection-evals, upstream commit `2f4399526773e2590f52490b633f479399407e80`.
- [Local Gleipnir 4B inference campaign preparation](plans/local_inference_throughput.md) (2026-09-23): Prepared 2026-09-23.
- [Possible monitoring training follow-ups](plans/monitoring_training_followups.md) (2026-09-08): fixed-backbone, existing-teacher-data proposals, their empirical and literature motivation, and a candidate comparison plan; research backlog rather than a launched campaign or selected recipe.
- [Action-only tool-trajectory monitoring](plans/tool_transcript_monitoring.md) (2026-08-30): operational paper notes, dataset provenance, and the proposed exploratory Kimi-logit scaling design for action-only agent monitoring.

## Writeups and history

- [Control Monitoring Has an Interface Problem](writeups/gleipnir_project_writeup.md): Simple logit distillation for stronger, cheaper monitors*
- [Mechanistic interventions for prompt injection sensitivity in logit-based trained monitors](writeups/mechanistic_injection_writeup.md) (2026-10-11): 11 October 2026*
- [Small Monitors on SLEIGHT-Bench](writeups/sleightbench_monitor_evaluation.md): How distilled action-only monitors fare against hand-built evasion attacks*
- [MATS project log](mats_project_log.md): append-only session log for the
  20-hour MATS sprint.
- [Migrations](migrations/aletheia_distillation_findings.md): material carried
  over from precursor projects.
