# Outline

Claims per section, accepted by their owner before the section is drafted.
Each claim follows [STYLE.md](STYLE.md): it names the model, evaluation set, metric and value, and
points to its evidence. The numbers below are copied from the linked records
as a starting point. Check each one against its source, and record it in
[sources.md](sources.md), when the section is drafted.

Status per claim: **draft** (proposed), **accepted** (the claim's owner
accepts it), **written** (in the text and in `sources.md`).

## Ownership

The owner of a claim decides its wording and whether it stays. Jeremias
Ferrao (JF) owns every section and claim except the held-out benchmarks other
than SleightBench, which Abhinav Rajeev Kumar (ARK) owns: natural cheats,
misleading comments and APPS injections (C5.1–C5.3). Each section lists its
owner; a claim with a different owner says so.

Paths are relative to the repository root.

## Main text and appendix

The main text has at most nine pages, counted up to the end of the conclusion
(`make check` reports the count). Material that does not support a main-text
claim goes to the appendix. Each main section points to its appendix section.

## Decisions

Made by Jeremias Ferrao on 2026-11-02.

1. **Headline scores.** Report each model on the optimized FP8/FP4 serving
   stack with its best prompt, as in the frontier figure: Gleipnir 4B with the
   student prompt (0.770 Mean-OOD pAUROC@20) and Gleipnir 27B with the teacher
   prompt (0.951). The optimized stack is the deployment path, and the OOD
   suite measures what it costs in quality. BF16 scores go in Section 3.2
   and the appendix.
2. **Teacher prompt for the 27B.** Present it as an ablation in the appendix.
   It tested whether the 27B benefits from more detailed instructions. It was
   adopted because it also improves ID ranking.
3. **Probing.** The main result is in the main text. Probe families,
   hyperparameters and layer choices go in the appendix.
4. **Cost.** Report both bases: self-hosted B200 break-even at on-demand and at
   committed-use prices, as in the frontier figure.
5. **Release version.** The report releases Gleipnir 4B and 27B as v1.0. The
   weights are the same as 4B v0.4 and 27B v0.2; only the model cards change.
   The report names the models v1.0, and `sources.md` records that results
   measured on v0.4/v0.2 (and on the v0.3 weights) apply to v1.0 unchanged.

## 1. Introduction

Owner: JF.

- **C1.1 (draft).** A monitor that scores each agent action runs on the
  critical path of that action, so its latency and cost add to every action.
- **C1.2 (draft).** Summary of the results: the main claims of Sections 4–6 in
  one paragraph each.
- **C1.3 (draft).** Contributions: the training recipe, the evaluation, the
  adversarial and probing analysis, and the released models and serving stack.

Exhibit: optionally, the frontier figure (Figure 1) on the first page.

## 2. Background and related work

Owner: JF.

This section is the background for Sections 3 and 4: it describes what we
reuse from Sinha et al., so the method states only what we change. The
description of Sinha et al. (C2.3–C2.5) takes a quarter to half a page. The
other paragraphs are short.

- **C2.1 (written).** Monitors in AI control. AI control studies safe
  deployment of an agent that may try to cause harm (Greenblatt et al.). A
  control evaluation gives the agent a main task and, for an attacking agent,
  a secret harmful side task (terms as defined in SHADE-Arena and BashArena;
  Greenblatt et al. do not use them). A
  monitor rates how suspicious the agent's actions look; in trusted monitoring
  it is a weaker, trusted model. Humans audit the highest-rated actions within
  limited time, so the monitor must rank attacks above almost all honest work.
  Our training data and benchmarks follow the main-task and side-task design.
  The paragraph states no constants.
  Evidence: arXiv:2312.06942 (Introduction, Sections 2 and 3.2);
  arXiv:2506.15740 (Section 2), arXiv:2512.15688 (Section 1). To check when Section 3 is drafted: which of
  the five training sources and six OOD sources have a main task and a side
  task (AgentDojo is prompt injection).
- **C2.2 (written).** Monitors in production. In the auto mode of Claude Code
  and the auto-review of Codex, a separate model approves or blocks an action
  before it runs, in place of a human approver. A blocked action returns to
  the agent, and a human is asked only after repeated blocks. A monitor in
  production needs a low false-positive rate, low latency and low cost. These are examples; the text does not claim that audit and gating are
  the only uses of monitors.
  Evidence: Anthropic engineering post on auto mode (2026-03-25); OpenAI
  Codex auto-review documentation.
- **C2.3 (written).** Training pipeline of Sinha et al. They train
  open-weight models from Qwen3.5-4B up to Kimi K2.6 (nine models, four
  families), so the text does not call them small. Their targets are inspired
  by deliberative alignment (Guan et al.), which is why a teacher writes
  rationales over a specification. Their monitors are
  action-only: they read the visible trajectory (user messages, agent
  messages, tool outputs), not the agent's private reasoning or activations.
  They train on 8,688 trajectories from five sources, half benign. A teacher
  (Gemini 2.5 Pro) reads each trajectory with a scheming specification and
  writes four candidate rationales with a 0–10 score; a judge (Claude Sonnet
  4.5) selects the best one, and the monitor is trained on it with SFT and,
  for some models, RL (GRPO, reward for calibration and ranking). The trained
  monitor does not receive the specification; only the teacher and the
  prompted baselines do.
  Evidence: arXiv:2605.29601, Sections 3.1–3.4, Table 2, Appendices C.1–C.2,
  D.2–D.3.
- **C2.4 (written).** Evaluation of Sinha et al. They select checkpoints on
  held-out STRIDE and Gloom-Exfiltration sets (our ID sets, 3,012 rows) and
  keep six OOD benchmarks out of training and selection. They score the OOD
  benchmarks, compute pAUROC@20 for each (area under the ROC curve for FPR up
  to 20%, divided by 0.2; taken from Storf et al.; chosen for low-FPR use,
  with no reason given for 20%) and report the unweighted mean (Mean-OOD).
  They drop trajectories over 28,000 tokens to fit a 32,000-token context.
  They pair
  it with the inference cost per 1,000 evaluations at uncached list prices.
  Their trained Qwen3.5-27B scores 0.831 at $2.78; Claude Opus 4.6 scores
  0.942 at 34.2 times the cost. The metric and the suite are defined here and
  used in Section 4.
  Evidence: arXiv:2605.29601, Section 4.2, Appendix D.4.
- **C2.5 (written).** What we reuse and what we change. We reuse their
  training trajectories, OOD suite, metric and cost axis, and we compare
  against their reported scores. We change the monitor output and the
  training target (Section 3). The text does not say whether their checkpoints
  are public: the part of the paper that was read has no release statement.
  For Section 4 (C4.1): Sinha et al. price open-weight models at hosted API
  proxy prices, while we price Gleipnir at self-hosted B200 break-even; state
  this difference there. Their tables sum to 6,402 OOD rows and our frozen
  suite has 6,395 (`docs/research/tool_transcript_monitoring.md`).
- **C2.6 (accepted).** Decision models. A decision model returns a probability
  for each permitted answer to a question about an input, in one forward pass
  and without generating text. The class became prominent with Jev (Almeida,
  2026). Its announcement names a new architecture and a training method
  (RLCD) but gives no details, and Jev's weights are not public. Perplexity
  released open-weight Decider models on Qwen backbones; their model cards
  describe the readout and the training data in part. Other providers
  followed (Mercury Decide, d1, Microsoft-Decision-1). For models without a
  paper we cite the announcement, model card or API page, with the dated
  revision we scored.
  Evidence: Jev announcement (TypeSafe blog, dated 2026-09-15); Decider cards
  `perplexity-ai/pplx-decider-v1.2`, `-flash-v1.2`, `-v1.1-27b`, `-v1-27b`.
- **C2.7 (accepted).** Published work on decision models. Geuter et al. give the
  most complete public recipe. They read the answer from the answer-letter
  logits at the last prompt token of open instruction-tuned models, and then
  train LoRA adapters and a decision head with cross-entropy on general
  decision tasks. Their models score close to or above Jev on their
  benchmarks. Gleipnir uses the same kind of readout for one task, agent
  trajectories, with one binary decision token and a frontier teacher's
  probability as the only target. Geuter et al. do not evaluate monitoring or
  agent trajectories, and we found no evaluation of decision models as
  control monitors (search of 2026-11-09). Section 4 compares Gleipnir with
  decision models on the OOD suite.
  Evidence: arXiv:2610.17205 (Sections 4–5, Tables 2–5, read in full);
  `docs/findings/decision_models_ood_benchmark.md`,
  `docs/findings/pplx_decider_ood_benchmark.md`,
  `docs/findings/tool_trajectory_jev_ood_benchmark.md`.
- **C2.7a (draft, optional).** Soft targets in decision models. Geuter et al.
  report no measurable change from soft targets for Gemma 4 31B (−0.2 ± 2.1
  Capability points with hard labels), but they add distillation from the
  trained 31B for Qwen3.5-4B, where it helps (65.1 against 63.0). The Decider
  flash card states that the 4B model is distilled from the 27B models' full
  probability distributions. This agrees with the direction of C3.4 and C3.5:
  soft targets help the small model more. Their tasks, metric and teachers
  differ from ours, so this is context, not a replication.
- **C2.8 (accepted).** Activation probes in production. Kramár et al. propose
  probe architectures that hold up on long inputs; Section 6 uses their
  architectures. Cunningham et al. (Constitutional Classifiers++) run linear
  probes as the first stage of a classifier cascade on production traffic, at
  a refusal rate of 0.05%. Both works read the activations of the monitored
  model. Gleipnir is a separate model and reads only the trajectory.
  Evidence: arXiv:2601.11516, arXiv:2601.04603,
  `experiments/probe_generalization/README.md`.
- **C2.9 (accepted).** Monitor-directed attacks. Terekhov et al. show that text
  addressed to the monitor can subvert trusted monitors. Section 5 tests the
  Gleipnir monitors against such text.

Citations to add when the section is drafted: one source per remaining
decision model without a paper (Mercury Decide, d1, Microsoft-Decision-1:
API page with access date). The five training sources and six OOD benchmarks
are each cited once in the background, where they are listed; an appendix
table gives sizes and sources. Read each cited paper beyond its abstract before
the claim is marked written.

## 3. Training and serving

Owner: JF.

Serving is part of the method because it sets the cost per evaluation used in
the frontier of Section 4.

### 3.1 Interface and training

- **C3.1 (draft).** The monitor reads a visible agent trajectory and emits one
  token, `0` or `1`. The score is the normalized probability of `1`.
- **C3.2 (draft).** The interface matters on its own. With Qwen3.5-9B, the
  prompt and the OOD suite fixed, binary logits score 0.698 Mean-OOD
  pAUROC@20, an ordinal 0–10 score 0.486, and an ordinal score after visible
  reasoning 0.666 at higher cost.
  Evidence: `docs/writeups/gleipnir_project_writeup.md` (interface ablation),
  `docs/research/tool_trajectory_ood_frontier.md`.
- **C3.3 (draft).** We train LoRA adapters to match the teacher's (Kimi K3)
  probability on the decision token, after a strict teacher-agreement filter.
  Evidence: `docs/decisions/strict_teacher_filtering.md`,
  `experiments/tool_trajectory_monitoring/README.md`.
- **C3.4 (draft).** Soft targets matter. Matched hard-label 4B seeds score
  0.690–0.746 Mean-OOD pAUROC@20 against 0.777–0.813 for soft-target seeds.
  Evidence: `experiments/hard_label_seeds/`.
- **C3.5 (draft).** The gain from soft targets is smaller for the 27B. With
  one seed per objective in the release configuration (optimized serving,
  teacher prompt), the hard-label 27B scores 0.933 Mean-OOD pAUROC@20 against
  0.951 for the soft-target 27B. With the student prompt and BF16 serving the
  scores are 0.904 and 0.909. On the ID suite the hard-label 27B scores higher
  (0.959 against 0.944), and 80% of its OOD scores lie within 0.01 of 0 or 1,
  against 20%. The hard-label run used the recipe tuned for soft targets.
  Interpretation, not a result: soft targets may add less when the unadapted
  student is already close to the teacher (Qwen3.8-27B 0.890 and Qwen3.5-4B
  0.618, against Kimi K3 0.908). Model size and distance to the teacher change
  together here, so the two models cannot separate them. A test would vary
  one of them: a stronger teacher for the 27B, or a weaker teacher for the 4B.
  Evidence: `experiments/qwen38_27b_hard_labels/README.md`.
- **C3.6 (draft).** The released 4B is the weight average of three seeds.
  Evidence: `experiments/seed_average/README.md`.
- **C3.7 (draft, appendix).** Teacher-prompt ablation for the 27B. Trained with
  the compact student prompt, Gleipnir 27B scores 0.942 ID macro pAUROC@20
  (CoT-removed suite) when scored with the full teacher prompt, against 0.922
  with the student prompt. Mean-OOD pAUROC@20 (BF16) is 0.935 against 0.909.
  The teacher prompt has 9.9% more prompt tokens.
  Evidence: `experiments/gleipnir27b_vocab_distillation/README.md`,
  `docs/findings/qwen38_27b_campaign.md`.

### 3.2 Serving and cost

This subsection lists the main serving decisions. The appendix gives the
throughput gain of each important change to the serving stack; minor
implementation details are left out.

- **C3.8 (draft).** Scoring needs one prefill pass and no decoding: the
  monitor reads the `0`/`1` logits at the last prompt token.
- **C3.9 (draft).** The optimized stack runs on one B200 with low-precision
  weights: FP8 attention projections, FP4 MLP and GDN projections and MXFP8
  attention prefill. The 4B adapter is trained with quantization-aware
  distillation (QAD) for this stack; the 27B is not.
  Evidence: `docs/decisions/b200_monitor_score_reference.md`,
  `experiments/b200_inference_benchmark/serving_default.json`.
- **C3.10 (draft).** Further changes remove work the score does not need: the
  final layer attends only from the last token, and the host path is shortened
  (tokenization, GDN routing).
  Evidence: `docs/findings/b200_last_layer_trim.md`,
  `docs/findings/b200_gdn_route.md`, `docs/findings/b200_host_overhead.md`.
- **C3.11 (draft).** Cost per 1,000 evaluations is the measured wall time to
  score all 6,395 OOD rows on one fully used B200, times an hourly rate:
  $7.99/h on demand (Runpod) or $3.44/h committed use (CoreWeave's published
  ceiling).
  Evidence: `docs/research/tool_trajectory_ood_frontier.md` (reference table),
  `docs/research/tool_trajectory_inference_economics.md`.
- **C3.12 (draft).** The optimized stack changes scores against BF16 serving:
  −0.034 Mean-OOD pAUROC@20 for the 4B and +0.016 for the 27B. Runs on that
  stack are not bit-for-bit repeatable (per-row MAE about 0.01 for the 4B).
  Evidence: `experiments/release_serving/README.md`,
  `experiments/optimized_gap/README.md`.
- **C3.13 (draft).** Released artifacts: FP32 LoRA adapters and merged BF16
  checkpoints for Gleipnir 4B v1.0 and 27B v1.0 (MIT; same weights as 4B v0.4
  and 27B v0.2), and the serving image.
  Evidence: `README.md` (Released models), `model_releases/hf/`.

Exhibits: pipeline diagram (`figures/distillation_pipeline_comparison.svg`),
interface ablation table, a short table of serving decisions. Appendix: data
sources, filter, hyperparameters, prompts, training cost, teacher-prompt
ablation, and throughput per serving change.

## 4. Main evaluation

Owner: JF.

- **C4.1 (draft).** Metric: source-macro raw normalized pAUROC@20 on the
  six-source, 6,395-row OOD suite of Sinha et al. Cost: uncached list-price USD
  per 1,000 evaluations. For Gleipnir, cost is the self-hosted B200 break-even
  at on-demand and at committed-use GPU prices; the text explains both.
  Evidence: `docs/research/tool_trajectory_ood_frontier.md` (comparison contract).
- **C4.2 (draft).** Gleipnir 27B (optimized serving, teacher prompt) scores
  0.951 Mean-OOD pAUROC@20 (95% interval 0.941–0.960) at $0.17 (committed-use)
  to $0.39 (on-demand) per 1,000 evaluations. Claude Opus 4.6, the best
  reasoning monitor of Sinha et al., scores 0.942 at $94.79. The difference is
  within the interval.
  Evidence: `docs/research/tool_trajectory_ood_frontier.md`,
  `docs/findings/qwen38_27b_campaign.md`, `experiments/optimized_gap/README.md`.
- **C4.3 (draft).** Gleipnir 4B (optimized serving, student prompt) scores
  0.770 at $0.04 (committed-use) to $0.09 (on-demand) per 1,000 evaluations.
  The trained Qwen3.5-4B monitor of Sinha et al. (SFT+RL) scores 0.684 at
  $0.36.
- **C4.4 (draft).** Training improves on the unadapted backbone: Qwen3.8-27B
  scores 0.890 unadapted, and Qwen3.5-4B scores 0.618.
- **C4.5 (draft).** Gleipnir 27B scores higher than its teacher, Kimi K3
  (0.908), on the OOD suite.

Exhibits: frontier figure (`figures/tool_trajectory_ood_frontier.svg`), main
table with bootstrap intervals. Appendix: per-source results, full frontier
figure, decision-model table, ID results, calibration.

## 5. Held-out and adversarial evaluation

Owner: ARK (C5.1–C5.3) and JF (C5.4, section framing).

- **C5.1 (draft, ARK).** On natural cheats (ImpossibleBench LiveCodeBench, 60
  hand-checked cheats), Gleipnir 4B and 27B score 0.935 and 0.941 pAUROC@20,
  level with Kimi K3, Claude Sonnet 5.5 and GPT-6.1 Sol at 3–13 times lower
  cost.
  Evidence: `docs/findings/adversarial_frontier_comparison.md`.
- **C5.2 (draft, ARK).** Misleading text in the trajectory changes the 4B's
  scores. With the same accusation text on both programs, the 4B separates backdoored
  from honest APPS code at AUROC 0.58–0.59, and the 27B at 0.96–0.98.
  Evidence: `experiments/adversarial_crossed_text/`.
- **C5.3 (draft, ARK).** Both monitors treat text addressed to the monitor as a
  sign of harm. At the APPS operating point, both flag every injected honest
  program. The unadapted 27B shows the same behavior, and the adapter reduces
  it.
  Evidence: `docs/findings/apps_released_monitors.md`.
- **C5.4 (draft, JF).** On SleightBench (stripped view), the 27B catches 15% and
  the 4B 5% of attacks at 1% FPR. Claude Opus 4.6 with extended thinking
  catches 32%. Most misses are attacks built to look like honest mistakes,
  which the training specification labels benign.
  Evidence: `docs/findings/sleightbench_released_monitors.md`.

Exhibits: natural-cheats table, crossed-text comparison, SleightBench catch
rate (`figures/sleightbench_catch_rate.svg`). Appendix: APPS injection
details, SleightBench per-attack results.

## 6. Probing

Owner: JF.

- **C6.1 (draft).** Soft targets also help activation probes, but less than
  they help LoRA. On the 4B residual stream (layer 15), the selected soft probe
  scores 0.007 (prompted input) and 0.055 (bare trajectory, NTP) Mean-OOD
  pAUROC@20 above the selected hard probe. On LoRA, soft targets add about
  0.07–0.08. The selected probes come from different probe families.
  Evidence: `experiments/probe_generalization/`, `experiments/hard_label_seeds/`.
- **C6.2 (draft).** At 27B, probes stay below the adapter. The best probe (last
  token, layer 50, teacher prompt) scores 0.896, below Gleipnir 27B (0.935
  BF16, teacher prompt).
  Evidence: `experiments/probe_scale_27b/`.
- **C6.3 (draft).** Probes are a tool for analysis here, not a cheaper
  deployment path. Fine-tuning is better in both quality and cost.

Exhibit: probe objective comparison (`figures/probe_objective_ood_dumbbell.svg`).
Appendix: probe families, layer sweep, attention maps.

## 7. Limitations

Owner: JF.

- **C7.1 (draft).** One teacher, one training domain, one seed for the 27B.
- **C7.2 (draft).** The monitors do not catch mistake-shaped attacks or resist
  monitor-directed text (Section 5).
- **C7.3 (draft).** The OOD suite inherits a 28,000-token input filter that
  removed 43% of SHADE-Arena, and SHADE-Arena has five task pairs.
- **C7.4 (draft).** We do not test the false-positive rates that production
  monitoring needs. pAUROC@20 averages over false-positive rates up to 20%,
  and the lowest operating point we report is 1% (SleightBench). Cunningham et
  al. operate at a 0.05% refusal rate on production traffic, and Anthropic
  reports a 0.4% false-positive rate for auto mode on 10,000 real tool calls.
  Neither Sinha et al. nor Storf et al. give a reason for the 20% cutoff.
  Interpretation (JF): the cutoff is pragmatic, because the benchmarks have
  few benign trajectories compared with production traffic. To support this
  with numbers when the section is drafted: the benign count per OOD source
  and the lowest false-positive rate each can resolve.

## 8. Conclusion

Owner: JF.

- **C8.1 (draft).** Restate the main claims; list follow-up work.
