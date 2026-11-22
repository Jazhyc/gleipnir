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
   suite measures what it costs in quality. BF16 scores go in Section 4
   and the appendix.
2. **Teacher prompt for the 27B.** Present it as an ablation in the appendix.
   It tested whether the 27B benefits from more detailed instructions. It was
   adopted because it also improves ranking on the validation set.
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
- **C1.2 (draft).** Summary of the results: the main claims of Sections 5–6 in
  one paragraph each.
- **C1.3 (draft).** Contributions: the training recipe, the evaluation, the
  adversarial and probing analysis, and the released models and serving stack.
  State the pricing decision here (JF, 2026-11-14): baselines are priced at
  standard API prices and Gleipnir at the cost of a dedicated self-hosted
  GPU, although the architectures match, because a prefill-only workload uses
  the GPU differently from decoding and a standard API price does not pass
  that on. Point to Section 4.1 and to the Decider list price as support.

Exhibit: optionally, the frontier figure (Figure 1) on the first page.

## 2. Background and related work

Owner: JF.

This section is the background for Sections 3 and 5: it describes what we
reuse from Sinha et al., so the method states only what we change. The
description of Sinha et al. (C2.3–C2.5) takes a quarter to half a page. The
other paragraphs are short.

Length: Section 2 stays near one page. Details of Sinha et al. that the
background leaves out are in Appendix A (written): the teacher, the four
candidates, the judge and the RL stage; the data table with counts per source;
the 28,000-token filter; the metric formula; and the cost basis. Section 2.2
points to it. The filter is also stated in limitation C7.3, and our own cost
basis in Section 5 (C4.1).

Subsections: 2.1 Monitoring agent actions (C2.1–C2.2), 2.2 Trained monitors
and their evaluation (C2.3–C2.5), 2.3 Related approaches (C2.6–C2.8).

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
  rationales over a specification. We keep the action-only restriction
  because reasoning and activations may become unreliable in future models:
  latent reasoning leaves nothing to read (Korbak et al.) and activations can
  be obfuscated (Bailey et al.). This is our assumption, stated as such.
  Their monitors are
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
  held-out STRIDE and Gloom-Exfiltration sets (the validation set, 3,012 rows) and
  keep six OOD benchmarks out of training and selection. They score the OOD
  benchmarks, compute pAUROC@20 for each (area under the ROC curve for FPR up
  to 20%, divided by 0.2; chosen for low-FPR use, with no reason given for
  20%; the text does not attribute the metric to Storf et al.) and report the unweighted mean (Mean-OOD).
  They drop trajectories over 28,000 tokens to fit a 32,000-token context.
  They pair
  it with the inference cost per 1,000 evaluations at uncached list prices.
  The metric and the suite are defined here and used in Section 5. The
  background gives no scores (their Qwen3.5-27B at 0.831 and $2.78, Opus 4.6
  at 0.942 and 34.2 times the cost, go in Section 5) and does not repeat why
  low false-positive rates matter (Section 2.1).
  Evidence: arXiv:2605.29601, Section 4.2, Appendix D.4.
- **C2.5 (written).** What we reuse and what we change. We reuse their
  training trajectories, OOD suite, metric and cost axis, and we compare
  against their reported scores. We change the monitor output and the
  training target (Section 3). The text does not say whether their checkpoints
  are public: the part of the paper that was read has no release statement.
  For Section 5 (C4.1): Sinha et al. price open-weight models at hosted API
  proxy prices, while we price Gleipnir at self-hosted B200 break-even; state
  this difference there. Their tables sum to 6,402 OOD rows and our frozen
  suite has 6,395 (`docs/research/tool_transcript_monitoring.md`).
- **C2.6 (written).** Decision models. A decision model returns a probability
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
- **C2.7 (written).** Published work on decision models. Geuter et al. give the
  most complete public recipe. They read the answer from the answer-letter
  logits at the last prompt token of open instruction-tuned models, and then
  train LoRA adapters and a decision head with cross-entropy on general
  decision tasks. Their models score close to or above Jev on their
  benchmarks. Gleipnir uses the same kind of readout for one task, agent
  trajectories, with one binary decision token and a frontier teacher's
  probability as the only target. Geuter et al. do not evaluate monitoring or
  agent trajectories, and we found no evaluation of decision models as
  control monitors (search of 2026-11-09). Section 5 compares Gleipnir with
  decision models on the OOD suite.
  Evidence: arXiv:2610.17205 (Sections 4–5, Tables 2–5, read in full);
  `docs/findings/decision_models_ood_benchmark.md`,
  `docs/findings/pplx_decider_ood_benchmark.md`,
  `docs/findings/tool_trajectory_jev_ood_benchmark.md`.
- **Revisions to C2.6–C2.7 (JF, 2026-11-09).** The text defines a decision
  model by its use of a pretrained language model with the permitted answers
  given in the request, not fixed in the architecture. It does not call Geuter
  et al. the most complete recipe; it says what they show. It cites Liu
  (arXiv:2609.33401), who evaluates decision models on security checks of an
  agent's inputs, and limits our claim to "no evaluation of decision models in an AI
  control scenario". It does not say "six" decision models: Section
  4 scores dedicated decision models and general language models read out in
  the same way.
- **Revision to C2.8 (JF, 2026-11-09).** The probes paragraph states two
  advantages of a probe on the monitored model: privileged access to its
  internal state, which may hold a stable representation of misaligned intent
  (Goldowsky-Dill et al.), and near-zero inference cost, since the probe
  reuses the model's own forward pass. Both need access to the monitored
  model, which the action-only setting does not assume. The paragraph does
  not point to the probing appendix, so that our probing experiments are not read as
  resting on these two advantages.
- **C2.7a (written in Appendix C.4 as one sentence, without the numbers).** Soft targets in decision models. Geuter et al.
  report no measurable change from soft targets for Gemma 4 31B (−0.2 ± 2.1
  Capability points with hard labels), but they add distillation from the
  trained 31B for Qwen3.5-4B, where it helps (65.1 against 63.0). The Decider
  flash card states that the 4B model is distilled from the 27B models' full
  probability distributions. This agrees with the direction of C3.4 and C3.5:
  soft targets help the small model more. Their tasks, metric and teachers
  differ from ours, so this is context, not a replication.
- **C2.8 (written).** Activation probes in production. Kramár et al. propose
  probe architectures that hold up on long inputs; the probing appendix uses their
  architectures. Cunningham et al. (Constitutional Classifiers++) run linear
  probes as the first stage of a classifier cascade on production traffic, at
  a refusal rate of 0.05%. Both works read the activations of the monitored
  model. Gleipnir is a separate model and reads only the trajectory.
  Evidence: arXiv:2601.11516, arXiv:2601.04603,
  `experiments/probe_generalization/README.md`.
- **C2.9 (moved to Section 6).** Monitor-directed attacks. Terekhov et al.
  show that an agent can evade monitors by embedding prompt injections in its
  outputs. This is not an approach related to ours, so Section 2.3 does not
  cover it. Section 6 introduces it where it tests the Gleipnir monitors
  against such text (C5.2, C5.3).

Citations to add when the section is drafted: one source per remaining
decision model without a paper (Mercury Decide, d1, Microsoft-Decision-1:
API page with access date). The five training sources and six OOD benchmarks
are each cited once in the background, where they are listed; an appendix
table gives sizes and sources. Read each cited paper beyond its abstract before
the claim is marked written.

## 3. Training

Owner: JF.

Structure (JF, 2026-11-10): serving and cost is its own Section 4, and
Section 3 is "Training Gleipnir monitors". The later sections moved up by one
(main evaluation 5, held-out 6, probing 7, limitations 8, conclusion 9). Claim
IDs keep their old numbers (C3.8–C3.13 are the serving claims, C4.x the main
evaluation, and so on).

Appendix order (JF, 2026-11-10): the prompts appendix is last (F), so results
and ablations are Appendix C and the subsections below are C.1 to C.5.
C.6 is weight averaging for the 4B (`app:ablation-average`, added 2026-11-10).

The hard-soft comparison reports only the compact instruction with BF16
serving (JF, 2026-11-10); the release-configuration 27B pair (0.933 against
0.951 OOD, 0.959 against 0.944 validation) is no longer in the report.

Planned (JF, 2026-11-10): a follow-up appendix subsection on why SHADE-Arena
is a problematic benchmark; the instruction table therefore has no SHADE-Arena
column. Material: 27B 0.588 (compact) against 0.730 (full specification) in
BF16 and 0.832 under optimized serving; released 4B 0.299 (BF16, compact); 4B
seeds 0.221 to 0.309; five task pairs. Not drafted.

Experiments run 2026-11-10 (JF): `experiments/gleipnir4b_full_specification/`
gives the 4B rows of Table `tab:instruction` (BF16 only; the optimized arm was
dropped), and `experiments/filter_margin_ood/` gives validation and Mean-OOD
for all five rows of Table `tab:filter` from one BF16 serving path. C.3 states
that the margin was chosen on the validation set. Appendix C.7 (serving
precision, Table `tab:precision`) is written.

Ablations live in Appendix D (JF, 2026-11-10): `app:ood-results` is now
"Results and ablations" with subsections D.1 all monitors on the OOD suite,
D.2 output format, D.3 teacher-agreement filter, D.4 soft targets against hard
labels (Table `tab:hard-soft`), D.5 instruction of the 27B. The main text keeps
one or two sentences for each, with a pointer: output format at the end of
3.1, filter in 3.2, 27B instruction and a paragraph "Effect of the soft
targets" at the end of 3.3. The former Section 4.2 no longer exists; the main
evaluation (now Section 5) holds only the frontier. The notes below that place
the hard-soft comparison in a results subsection are superseded.

Subsections in the text: 3.1 Monitor input and output (C3.1–C3.2, written),
3.2 Soft-target distillation (C3.3, written), 3.3 Training setup (C3.6–C3.7
and the configuration). The comparison of hard labels and soft targets
(C3.4–C3.5, written) is in Appendix C.4 with Table `tab:hard-soft`. Section
3.2 does not report the false-positive rate of the filter: it is not the
metric of Sinha et al., and we have not measured its variation across seeds
(JF, 2026-11-10). The report says nothing about calibration for now, and
Table `tab:hard-soft` has no extreme-score column. How to report the
false-positive rate is undecided; one option is a second frontier plot. The
numbers and a first plot exist: `experiments/ood_recall_at_fpr/` (recall at 1%
FPR for our 21 monitors; Gleipnir 27B 0.874, Gleipnir 4B 0.606) and
`figures/ood_recall_at_1pct_fpr_frontier.svg`. They are not in the report.

Facts checked against the repository on 2026-11-09, for 3.2 and 3.3:

- Released 4B (v0.4) and 27B (v0.2) train only on the 7,836 trajectories left
  by the strict filter (852 of 8,688 removed). The 13,149 deception rows
  (Aletheia's Quest, Liars' Bench) were used only for the 9B and the 4B v0.1.
- Loss: binary cross-entropy between the monitor's score (Equation 1) and the
  teacher's probability; no hard-label term, no temperature.
- Teacher cache: Kimi K3 through OpenRouter, full specification, temperature
  zero, reasoning off; probability of `1` among the `0` and `1` logprobs;
  about $224 for the 8,688 rows (a lower bound on the spend).
- LoRA rank 128, alpha 256, no dropout, on attention and MLP projections;
  unquantized BF16 base with an FP32 adapter (not QLoRA); AdamW, learning
  rate 5e-5, 3% warmup then linear decay, batch 32, one epoch of 245 updates.
  One B200: about one hour per 4B seed and 4.5 hours for the soft-target 27B
  (`docs/findings/qwen38_27b_campaign.md`; the hard-label 27B took 4.1).
- 4B: three seeds that differ in data order, averaged exactly into one
  rank-384 adapter; then QAD against its own BF16 scores under the optimized
  serving quantization.
- Two instructions: a compact one (133 tokens) used for training and for the
  4B, and the full specification (701 words) used for the teacher and for the
  released 27B.
- **Data properties, written in the appendix (`app:data-discrepancies`).** The
  training inputs are not fully action-only: 1,340 of the 1,350 STRIDE
  training rows contain the agent's reasoning, and the Kimi K3 targets were
  computed on them. The validation set has reasoning removed. In five of the
  six OOD benchmarks the inputs lack the agent's own text and keep only tool
  calls and results. These are properties of the files that Sinha et al.
  prepared. Still to do: one sentence in 3.3 and one in the limitations that
  point to this appendix, and decide whether Section 2.2's "we keep this
  restriction" needs a qualifier.
- **Prompts, written in the appendix (`app:prompts`).** The compact
  instruction, the full specification and the ControlArena instruction are
  printed in full from byte copies under `assets/prompts/`.

### 3.1 Monitor input and output

- **C3.1 (written).** The monitor reads a visible agent trajectory and emits one
  token, `0` or `1`. The score is the normalized probability of `1`.
- **C3.2 (written).** The interface matters on its own. With Qwen3.5-9B, the
  prompt and the OOD suite fixed, binary logits score 0.698 Mean-OOD
  pAUROC@20, an ordinal 0–10 score 0.486, and an ordinal score after visible
  reasoning 0.666 at higher cost.
  Evidence: `docs/writeups/gleipnir_project_writeup.md` (interface ablation),
  `docs/research/tool_trajectory_ood_frontier.md`.
- **C3.3 (written).** We train LoRA adapters to match the teacher's (Kimi K3)
  probability on the decision token, after a strict teacher-agreement filter.
  Evidence: `docs/decisions/strict_teacher_filtering.md`,
  `experiments/tool_trajectory_monitoring/README.md`.
- **C3.4 (written).** Soft targets matter. Matched hard-label 4B seeds score
  0.690–0.746 Mean-OOD pAUROC@20 against 0.777–0.813 for soft-target seeds.
  Evidence: `experiments/hard_label_seeds/`.
- **C3.5 (written).** The gain from soft targets is smaller for the 27B. With
  one seed per objective in the release configuration (optimized serving,
  teacher prompt), the hard-label 27B scores 0.933 Mean-OOD pAUROC@20 against
  0.951 for the soft-target 27B. With the student prompt and BF16 serving the
  scores are 0.904 and 0.909. On the validation set the hard-label 27B scores higher
  (0.959 against 0.944), and 80% of its OOD scores lie within 0.01 of 0 or 1,
  against 20%. The hard-label run used the recipe tuned for soft targets.
  Interpretation, not a result: soft targets may add less when the unadapted
  student is already close to the teacher (Qwen3.8-27B 0.890 and Qwen3.5-4B
  0.618, against Kimi K3 0.908). Model size and distance to the teacher change
  together here, so the two models cannot separate them. A test would vary
  one of them: a stronger teacher for the 27B, or a weaker teacher for the 4B.
  Evidence: `experiments/qwen38_27b_hard_labels/README.md`.
- **C3.6 (written).** The released 4B is the weight average of three seeds
  that differ in data order, release rule not mentioned (JF, 2026-11-10)
  (validation 0.912 against a seed mean of 0.903), then trained with QAD.
  The QAD scores go in Section 4.
  Evidence: `experiments/seed_average/README.md`.
- **C3.7 (written in 3.3 with the validation scores only; the OOD scores
  0.935 against 0.909 are not in the text yet).** Teacher-prompt ablation for the 27B. Trained with
  the compact student prompt, Gleipnir 27B scores 0.942 validation macro pAUROC@20
  (CoT-removed validation set) when scored with the full teacher prompt, against 0.922
  with the student prompt. Mean-OOD pAUROC@20 (BF16) is 0.935 against 0.909.
  The teacher prompt has 9.9% more prompt tokens.
  Evidence: `experiments/gleipnir27b_vocab_distillation/README.md`,
  `docs/findings/qwen38_27b_campaign.md`.

## 4. Serving and cost

Owner: JF. File `sections/serving.tex`, label `sec:serving`. Written: the
central table (`tab:serving`, USD per million prompt tokens, all self-hosted
cells from the 320-trajectory benchmark) and 4.1 Self-hosting with one
forward pass (`sec:serving-selfhost`) and 4.2 Low-precision arithmetic
(`sec:serving-precision`) and 4.3 Removing unneeded work (`sec:serving-work`:
last-token attention in the final layer, shorter GDN host path; the
length-gated GDN variant of the 4B is in Appendix E.1). Section 4 text is
complete (JF, 2026-11-14). Released artifacts are not a subsection here; they
go in the abstract or introduction. Appendix E.1 has the environment, the
per-step table (`tab:serving-steps`) and the latency figure
(`fig:serving-latency`) from the EU-RO-1 session of 2026-11-15. Table 1
takes the 27B gains from that session (2.21x, 1.04x; JF, 2026-11-15); its
other cells keep the earlier measurements. The two paragraphs of 4.3 give
per-change gains only for the 4B (JF, 2026-11-15): the 27B changes were not
measured separately in that session.
Serving benchmark (JF, 2026-11-15/16): all serving numbers (Table 1, Section 4, Appendix E.1) now use 96 agent trajectories (BashArena, BashBench, STRIDE, Gloom; mean 12,430 tokens) sent one at a time (8 concurrent as the saturation case), measured on one machine; Table 1 costs follow from the one-request throughput and include Gigatoken, which Section 4.3 now describes. TODO (JF): regenerate the frontier figures and the costs of `tab:ood-all`, which still use the batched OOD-suite throughput (about 10% lower cost for the 4B than Table 1); the old 320-prompt benchmark was 70% short deception prompts. 4B graphs go up to 4,096 tokens. The notes below predate this and give 320-prompt figures. B200 breakdown session (2026-11-15, `experiments/serving_concurrency/README.md`, second Result): measured; in Appendix E.1. 27B per change: low precision 2.22×, larger graphs 1.00× (one-request p50 58.6 → 24.0 ms), last-token attention 1.01×, GDN host path 1.04×. 4B with graphs up to 2,048: one-request p50 29.0 → 11.3 ms, throughput and per-change gains unchanged, full concurrency curve measured. Open before any report change: with the larger graphs, 4B scores change for prompts of 257–767 tokens (OOD 0.7669 → 0.7639 at one request at a time, AgentDojo 0.8152 → 0.7973). Cause: below 768 rows the scores depend on whether the row count is a multiple of 32. A layout with sizes 32k − 1 and 32k (`fast_g2048e`) is bit-identical to the release on ID, OOD and the benchmark at one request at a time, with p50 11.6 ms; its concurrency curve is measured. Adopted for the report's Table 13, Figure and E.1 text (JF, 2026-11-15); the release image and command still need the update. Gigatoken (JF, 2026-11-15): it was inert on vLLM 0.31; added as the last serving step of Table 13 and to both panels of the figure from sweeps with a working hook (one request at a time 1.17× for the 4B, 1.05× for the 27B; no change at 128); the release image needs this hook too. Also new: the released 4B scores AgentDojo 0.8425 in batches of 128 and 0.8152 at one request at a time.
Appendix E.2 (`app:serving-kernels`) lists the kernels, their NVIDIA sources
and our changes; 4.2 points to it. Layer counts belong in the subsection on
removing unneeded work (JF, 2026-11-14): Qwen3.5-4B has 32 layers (8 full
attention, 24 GDN) and Qwen3.8-27B has 64 (16, 48), so last-token attention
covers 1 of 8 attention layers in the 4B and 1 of 16 in the 27B
(`docs/research/qwen38_27b_serving_transfer.md`). No backward-pass or training-stack detail
(JF, 2026-11-14: an implementation detail that users do not need).
Serving has its own section because it is a separate contribution and sets
the cost per evaluation used in the frontier of Section 5. Planned
subsections: the serving stack (changes against stock vLLM), the effect on
scores (BF16 against optimized, QAD for the 4B; numbers in Appendix C.7; one
clause citing `ouyang2025lowbit` for why the 27B needs no QAD), and cost and
released artifacts. Target about one page.

Central table (JF, 2026-11-10): one table that the subsections refer to,
with rows by category and columns for each model: hosted list price;
self-hosted stock vLLM BF16, prefill only; low-precision arithmetic; work the
score does not need (last-token final-layer attention, host path). Each
category is reported as a gain factor because the two ladders were measured
in different orders (4B: 1.77× then 1.073×; 27B: 1.02× then 2.24×). Speed and
cost only; scores stay in Appendix C.7. Factors come from the 320-row
benchmark; final costs from the measured OOD wall time. The appendix gives
the per-model breakdown. Caption states full utilization.

This section lists the main serving decisions. The appendix gives the
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
  The repeatability note is left out of the report (JF, 2026-11-14: hardware
  non-determinism is expected).
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

## 5. Main evaluation

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
table with bootstrap intervals. Appendix: the table of all OOD results
(written, `app:ood-results`: the 17 rows of Sinha et al. and our 21 measured
rows, including the decision models and GPT-5.6 Luna, without our GPT-OSS-120B
rows; move the section to match where the main
text first cites it), per-source results, full frontier figure, validation
results, calibration.

## 6. Held-out and adversarial evaluation

Owner: ARK (C5.1–C5.3, C5.5) and JF (C5.4, section framing).

- **C5.1 (written, ARK; 2026-11-20).** On reward hacking by 15 agent models
  (EvilGenie Escalation, 105 hacks and 999 legitimate runs both monitors
  scored), Gleipnir 27B reaches 0.695 pAUROC@20 and Gleipnir 4B 0.350; at the
  OOD 1% threshold they flag 33 and 10 of 105 hacks and both flag 8 of 999
  legitimate runs. One problem holds 69 hacks that keep the program correct;
  neither monitor flags them (0 and 4 of 69). Post hoc split. Main text 6.1.
  Evidence: `experiments/adversarial_evilgenie/` and its `shared_rows_report.py`.
- **C5.2 (written, ARK; 2026-11-20).** With the same module docstring on an
  honest and a backdoored APPS program, Gleipnir 4B falls from 0.690 to 0.143
  pAUROC@20 under a specific accusation, while Gleipnir 27B rises from 0.818 to
  0.934; fine-tuning made the 4B more sensitive than its base (AUROC 0.581
  against 0.805). Thresholds set on code without text flag 93-100% of accused
  honest programs. Main text 6.2.
  Evidence: `experiments/adversarial_crossed_text/`, `results/figures/backdoor_pauroc.json`.
- **C5.3 (written, ARK; moved to the appendix).** Both monitors treat text
  addressed to the monitor as a sign of harm (APPS injections: every injected
  honest program flagged at the operating point; lab post: 4B 82% of readers,
  27B and Kimi 0%). Appendix F.3 and F.8.
  Evidence: `docs/findings/apps_released_monitors.md`, `experiments/adversarial_is_it_4b/`.
- **C5.5 (written, ARK; appendix).** Natural cheats (ImpossibleBench, 57
  cheats): Gleipnir 4B and 27B score 0.935 and 0.941 pAUROC@20, level with
  Kimi K3, Claude Sonnet 5.5 and GPT-6.1 Sol. Comment removal, multi-agent
  sabotage (UK AISI benchmark) and agents lying to agents are in the same
  appendix. Costs are not compared there (mixed price bases in the records).
  Evidence: `docs/findings/adversarial_frontier_comparison.md` and the
  experiment READMEs listed in `sources.md`.
- **C5.4 (draft, JF).** On SleightBench (stripped view), the 27B catches 15% and
  the 4B 5% of attacks at 1% FPR. Claude Opus 4.6 with extended thinking
  catches 32%. Most misses are attacks built to look like honest mistakes,
  which the training specification labels benign.
  Evidence: `docs/findings/sleightbench_released_monitors.md`.

- **Decision (JF, 2026-11-22): the four smaller tests (natural
  cheats, comment removal, multi-agent sabotage, agents that lie to agents)
  are exploratory and have their own appendix section.**
  Their reference monitors and readouts differ by test; no main claim rests
  on them, and they are not rerun. The reruns below cover 6.2 and 6.3 only.
- **TODO (JF, 2026-11-21): rerun ARK's main held-out experiments on the
  released serving stack** once JF has reviewed them, so they match the rest
  of the report. Those runs used BF16 Transformers on an A100/H100 without the
  `flash-linear-attention` kernels, and the EvilGenie 27B run stopped at a
  pre-registered 2-hour cap after 1,312 of 2,295 rows. Score every row, and
  take the 1% thresholds from the released-stack OOD scores. Needs a B200 Pod
  (JF's go-ahead; budget).
- **TODO (JF, 2026-11-21; revised 2026-11-22): add baselines to the rerun.**
  Score, on the same rows of EvilGenie (6.2) and APPS (6.3), the unadapted
  Qwen3.5-4B and Qwen3.8-27B (same Pod), pplx-decider v1.3 and GPT-6 Luna as
  cheap references, and Kimi K3 (the teacher) as the high end, all with
  binary readouts; each already has OOD scores for its 1% threshold. Report
  Kimi K3's own agent runs in EvilGenie separately. GPT-6.1 Terra is not
  used (JF, 2026-11-22). The ordinal APPS monitors (GPT-6.1 Sol, Claude
  Sonnet 5.5, 0-100 at low reasoning effort) are dropped or kept only as a
  labelled reasoning reference.
  The 999 legitimate runs of 6.2 are an artefact of the 27B's cap (1,823
  labelled legitimate, 1,695 fit in context); the rerun should report the
  full set, with rates alongside counts.
  API budget (2026-11-22, list prices, input only; output is one token).
  Tokens per monitor: EvilGenie about 20M (2,113 runs that fit; ARK's 18M
  estimate plus the full specification), APPS about 1.5M (982 rows, after
  ARK's $0.81 Kimi spend). Kimi K3: Makora $1.00/M, about $22 with its
  existing OOD scores; Wafer (mxfp4, logprobs) $0.19/M, about $4, or about
  $6 if a parity canary against Makora forces a new OOD threshold (about
  9.5M tokens). GPT-6 Luna $0.10/M, about $2.2; pplx-decider v1.3 (free
  endpoint; $0.02/M list) under $0.50. Total about $8-25, plus about an hour
  of one B200 for the Gleipnir and base-model scoring.
  After the rerun, Appendix F.2 (protocol of the stress tests: BF16
  Transformers, separate OOD thresholds) should match the main setup; cut it
  to what still differs, or remove it.

Exhibits: natural-cheats table, crossed-text comparison, SleightBench catch
rate (`figures/sleightbench_catch_rate.svg`). Appendix: APPS injection
details, SleightBench per-attack results.

## 7. Probing (moved to the appendix)

Owner: JF.

Moved to the appendix (`app:probing`) on 2026-11-18 (JF): probing is
tangential to describing Gleipnir. The main text keeps two pointers: one
sentence in Section 3.2 that soft targets also help probes, and early exits
as future work in the conclusion. Sections 8 and 9 move up by one.

Framing (JF, 2026-11-09): the probing experiments ask whether early exits
could lower inference cost further. If the backbone of the monitor already
holds a representation that supports the decision before the final layer, a
probe at that layer could replace the remaining layers. The experiments also
ask whether soft distillation helps such probes. They do not assume access to
the monitored agent: the probes read the monitor backbone on the trajectory.
The probing appendix introduces Kramár et al.'s architectures itself; Section 2.3 does
not point here. The claims below predate this framing and need a pass against
it before they are accepted.

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

## 8. Limitations

Owner: JF.

- **C7.1 (draft).** One teacher, one training domain, one seed for the 27B.
- **C7.2 (draft).** The monitors do not catch mistake-shaped attacks or resist
  monitor-directed text (Section 6).
- **C7.3 (draft).** Sinha et al. drop every trajectory over 28,000 tokens to
  fit the 32,000-token context of their shorter-context models. The limitation
  states this, since the background does not. The OOD suite inherits the
  filter, which removed 43% of SHADE-Arena, and SHADE-Arena has five task
  pairs.
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
- **C7.5 (draft; JF, 2026-11-14).** The cost comparison mixes price bases.
  Baselines are priced at standard API prices, which is how monitors that
  write a rationale are used today, and Gleipnir at the cost of a dedicated
  self-hosted GPU (Section 4.1). A base model read with binary logits and
  self-hosted would cost the same as the Gleipnir monitor of that size, so
  the cost saving comes from the output format and serving and not from the
  training. A fairer comparison of open models measures interactive latency
  and throughput on the same hardware; state this as future work.

## 9. Conclusion

Owner: JF.

- **C8.1 (draft).** Restate the main claims; list follow-up work.
