# Sources and scope

Record each reported number's source document or artifact, the repository
revision it was read at, the metric definition and any comparison limits.
Add an entry here when you add a number to the report.

No results are reported yet.

## External references

- Terekhov et al., *Adaptive Attacks on Trusted Monitors Subvert AI Control
  Protocols*, [arXiv:2510.09462](https://arxiv.org/abs/2510.09462).
- Sinha et al., *Training Deliberative Monitors for Black-Box Scheming
  Detection*, [arXiv:2605.29601](https://arxiv.org/abs/2605.29601). Source of
  the training trajectories, the OOD suite, pAUROC@20 and the cost axis.
  Read on 2026-11-09 (v1, HTML): abstract to Appendix D.4 verbatim; Appendices
  E to K only as a machine summary or not at all. Read verbatim: monitor input
  (Section 3.1, Appendix C.1), training counts (Table 2), teacher, judge and
  four candidates (Appendix D.2), RL stage (D.3), pAUROC@20 and Mean-OOD
  (D.4), and 0.831 at $2.78 with Opus 4.6 +0.111 at 34.2 times the cost
  (Section 4.2). From the summary only: the cost definition (Appendix E.2) and
  LoRA rank 32 (G.1). The paper states that the trained monitor is not
  prompted with the specification. Length filter (Appendix C.1): trajectories
  over 28K tokens are dropped to leave 4K of headroom in the 32K context of
  the shorter-context models; the tokenizer and the number of removed rows are
  not stated. Metric (Section 3.4, Appendix D.4): pAUROC@20 "following Storf
  et al. (2026)" ([arXiv:2603.00829](https://arxiv.org/abs/2603.00829); read
  by JF on 2026-11-09, who reports that it also gives little detail on the
  choice of cutoff), chosen because low false-positive rates matter for deployment
  and human review. The paper gives no reason for the 20% cutoff itself.
- Kramár et al., *Building Production-Ready Probes For Gemini*,
  [arXiv:2601.11516](https://arxiv.org/abs/2601.11516). Source of the probe
  architectures in Section 7.
- Cunningham et al., *Constitutional Classifiers++*,
  [arXiv:2601.04603](https://arxiv.org/abs/2601.04603). The 0.05% figure is
  the refusal rate on production traffic stated in the abstract.
- Greenblatt et al., *AI Control: Improving Safety Despite Intentional
  Subversion*, [arXiv:2312.06942](https://arxiv.org/abs/2312.06942), v5, read
  2026-11-09 up to Section 3.2. Trusted monitoring and the 2% audit budget:
  Introduction, Sections 2.2 and 3.2, Table 2. Failure on one submitted
  backdoor: Section 2. In their setting the monitor reads the problem and the
  code. How the audit threshold is set (Appendix G) was not read.
- Geuter et al., *Releasing the Inner Jev*,
  [arXiv:2610.17205](https://arxiv.org/abs/2610.17205), v1, HTML version read
  in full on 2026-11-09. Soft-against-hard ablation: Table 5; Qwen3.5-4B
  distillation: Section 4 and Appendix B.
- Almeida, *Introducing System One Models & Jev*, TypeSafe AI blog,
  `https://typesafe.ai/blog/introducing-system-one-models-and-jev`, read
  2026-11-09. The post shows the date Sep 15, 2026; its page metadata says
  Nov 6, 2026, which we take to be a later edit. Claims in it are
  self-reported.
- Perplexity Decider model cards under `https://huggingface.co/perplexity-ai/`,
  read 2026-11-09: `pplx-decider-v1.2` (joint readout at a marker token,
  Qwen3.8-27B, Apache-2.0), `pplx-decider-flash-v1.2` (Qwen3.5-4B, distilled
  from the 27B models), `pplx-decider-v1.1-27b` (noncausal attention, data
  sources) and `pplx-decider-v1-27b`. The cards give no release dates, no
  paper and no citation for the models. The v1.2 cards call the repository
  private although the pages load without a token.
- A web search on 2026-11-09 found no survey of decision models and no
  evaluation of them as control monitors.
- Anthropic, *How we built Claude Code auto mode*, Anthropic Engineering blog
  (no byline; the acknowledgments say "Written by John Hughes"),
  `https://www.anthropic.com/engineering/claude-code-auto-mode`, published
  2026-03-25, read in full on 2026-11-09. A classifier reviews shell commands,
  web fetches, external tools and similar actions before they run; read-only
  tools and in-project edits skip it. It reads user messages and tool calls
  only. A denial returns to the agent; after repeated denials the session
  escalates to the human. Reported rates (March 2026 classifier): 0.4%
  false-positive rate on 10,000 real internal tool calls and 17% miss rate on
  52 real overeager actions. Its first stage is a single-token yes/no filter.
- OpenAI, *Auto-review*, Codex documentation,
  `https://developers.openai.com/codex/concepts/sandboxing/auto-review`
  (redirects to `learn.chatgpt.com`), undated, read 2026-11-09. A reviewer
  agent decides on actions that would otherwise need user approval; actions
  the sandbox already allows are not reviewed. Details beyond the opening of
  the page (denial handling, the limit on repeated denials) were read from a
  machine summary of the page and should be checked before they are quoted.
  The page gives no error rates.
- Main task and side task. A full-text check on 2026-11-09 found that
  Greenblatt et al. (arXiv:2312.06942) and Ctrl-Z (Bhatt et al.,
  arXiv:2504.10374) use neither term; they write of solutions, backdoors and
  attack policies. The terms are defined in Kutasov et al., *SHADE-Arena*,
  [arXiv:2506.15740](https://arxiv.org/abs/2506.15740), Section 2, and in
  Kaufman et al., *BashArena*,
  [arXiv:2512.15688](https://arxiv.org/abs/2512.15688), Section 1. Neither
  attributes them to an earlier source.
- Benchmark citations follow the dataset cards of Sinha et al. (Appendix C.3):
  STRIDE and Gloom (Storf et al.), CoT-Red-Handed (Arnav et al.,
  arXiv:2505.23575), BashArena (Kaufman et al.), BashBench (Bhatt et al.,
  arXiv:2504.10374), AgentDojo (Debenedetti et al., NeurIPS 2024 Datasets and
  Benchmarks, arXiv:2406.13352), CUA-SHADE-Arena (Kale et al.,
  arXiv:2508.19461), IaC (ControlArena web page), MLE-Sabotage (Ward et al.,
  arXiv:2511.09904), Rogue Eval (Roger et al., Anthropic Alignment Science
  blog) and SHADE-Arena (Kutasov et al.). The arXiv entries were checked
  against the arXiv API on 2026-11-09. The IaC and Rogue Eval entries are
  copied from the reference list of Sinha et al. and were not opened; the
  first names of the Rogue Eval authors are ours and need a check.
- Splits of Sinha et al. (Table 1, Table 2, Appendix C.1): training on the
  five sources; validation on STRIDE (test) and Gloom-Exfiltration, "used for
  checkpoint selection and in-domain held-out analysis" (369/577 and
  1,035/1,031 benign/scheming, 3,012 in total, which is the size of the
  set our project records call the ID suite); the six OOD benchmarks "are not used for training or checkpoint
  selection". That the OOD benchmarks cover environments absent from the
  training set (computer use, infrastructure code, ML research) is our
  reading of the dataset descriptions (training: enterprise software,
  operational tasks, coding sabotage, Linux administration); the paper's own
  definition of OOD is only that the sets are not used for training or
  checkpoint selection.
- Guan et al., *Deliberative Alignment: Reasoning Enables Safer Language
  Models*, [arXiv:2412.16339](https://arxiv.org/abs/2412.16339); abstract
  read 2026-11-09. Sinha et al. state in their abstract that their method is
  "inspired by deliberative alignment"; which reference they cite for it was
  not checked. Model range of Sinha et al.: Section 4.4 ("nine open-weight
  models belonging to four different model families") and the Figure 6
  caption ("from Qwen3.5-4B to Kimi-K2.6").
- Rationale for action-only monitors. Korbak et al., *Chain of Thought
  Monitorability*, [arXiv:2507.11473](https://arxiv.org/abs/2507.11473), and
  Bailey et al., *Obfuscated Activations Bypass LLM Latent-Space Defenses*,
  [arXiv:2412.09565](https://arxiv.org/abs/2412.09565); abstracts read
  2026-11-09. Korbak et al.'s abstract says CoT monitorability "may be
  fragile"; that the paper names latent reasoning as a cause is from memory
  and needs a check in the full text. Bailey et al. obtain obfuscated
  activations through attacks, not from models that obfuscate on their own.
  Sinha et al. give a related reason in their abstract: reasoning and
  activations "can be unavailable, unreliable or expensive in deployment".
- Appendix A. Pipeline details: Sinha et al., Sections 3.1–3.4 and Appendices
  D.2–D.3 (read verbatim): four candidates, the boundary at a score of 5,
  regeneration with label guidance, the judge and its threshold of 7, GRPO
  with a Brier-style term and a pairwise ranking term, and that the trained
  monitor is not prompted with the specification. Counts per source:
  `docs/research/tool_transcript_monitoring.md` (dataset inventory), which
  takes the training and validation counts from the paper's Table 2 and the
  OOD counts (6,395) from the authors' prepared inputs
  (`adityaasinha28/deliberative-monitor-pipeline@2f10f4e`). Domain labels are
  shortened from the same inventory. Cost basis: Appendix E.2 and the Table 3
  caption, read only as a machine summary.
- Table of all OOD results (`app:ood-results`). Every row is copied from the
  reference table in `docs/research/tool_trajectory_ood_frontier.md` (read at
  the merge commit of 2026-11-09), with costs and scores rounded to three
  decimals. The 17 rows of Sinha et al. reproduce their Appendix E, Table 4,
  as recorded in that registry; they were not rechecked against the paper on
  2026-11-09. Rounding was checked against full-precision values where the
  registry's fourth decimal is 5: unadapted Qwen3.5-4B 0.617500 (0.618),
  Qwen3.5-27B 0.839474 (0.839), GPT-6 Luna 0.836544 (0.837) at $0.835498
  (0.835), Jev 1.13 0.849513 (0.850). Gleipnir 4B and 27B are the optimized
  serving scores (student prompt and teacher prompt), priced at $3.44/h and
  $7.99/h for one B200. "Gleipnir 9B (earlier release)" is the registry row
  "Qwen3.5-9B Kimi-soft mixed". Results measured on 4B v0.3/v0.4 and 27B v0.2
  weights apply to v1.0 unchanged. Changes from the registry (JF,
  2026-11-09): our two unadapted GPT-OSS-120B rows are left out, because the
  model needs a reasoning field and the comparison with models that do not
  reason is unclear; GPT-5.6 Luna is added from
  `docs/findings/openai_56_luna_ood_benchmark.md` (0.839729, and $1.672442
  per 1,000 with all input at uncached standard rates), although the registry
  excludes it. The GPT-OSS rows that Sinha et al. report stay in the table.
- Section 2.3. Read only as abstracts (2026-11-09): Kramár et al.
  (arXiv:2601.11516; "new probe architectures that handle this long-context
  distribution shift"), Cunningham et al. (arXiv:2601.04603; linear probes,
  a two-stage classifier cascade, 0.05% refusal rate on production traffic)
  and Terekhov et al. (arXiv:2510.09462; prompt injections embedded in model
  outputs evade monitors on two AI control benchmarks). "Six" decision models
  are the six rows of the results table; Mercury Decide, d1, d1-ultra and
  Microsoft-Decision-1 still have no cited source.
- Liu, *Evaluating System One Models for Agent Security Decisions*,
  [arXiv:2609.33401](https://arxiv.org/abs/2609.33401), v3; abstract read
  2026-11-09. It evaluates Jev, Laya, Decider and Bespoke Nimble on
  prompt-injection detection, interaction-risk judgment and harmful-request
  screening, which are checks "before agents act". Three further web searches
  on 2026-11-09 for decision models as control monitors (sabotage, scheming,
  SHADE-Arena, BashArena, ControlArena) found no such evaluation; they also
  found community guardrail projects that gate tool calls with Jev, without
  published evaluations.
- Goldowsky-Dill et al., *Detecting Strategic Deception Using Linear Probes*,
  [arXiv:2502.03407](https://arxiv.org/abs/2502.03407); abstract read
  2026-11-09. Cited for the possibility that activations represent deceptive
  or misaligned intent (probe AUROC 0.96 to 0.999 on Llama-3.3-70B). The
  authors add that "current performance is insufficient as a robust defence
  against deception", so the text says "may hold". That a probe's cost is near
  zero because it reuses the monitored model's forward pass is our statement.
- Section 3.1. Input format, decision tokens and score:
  `experiments/tool_trajectory_monitoring/prompts/manifest.json` and
  `prompting.py`, `src/gleipnir/evaluation/probabilities.py`, and the 4B model
  card. Output-format comparison on unadapted Qwen3.5-9B (6,395 OOD rows):
  binary logits 0.6976 (`docs/findings/tool_trajectory_qwen_ood_benchmark.md`),
  0–10 integer 0.4858 and binary rounded to tenths 0.6835, a difference of
  0.0141 (`tool_trajectory_qwen_ordinal_ood_benchmark.md`), reasoning then
  integer 0.6656 and reasoning then binary 0.6923 (the two
  `..._reasoning_ood_benchmark.md` findings). Fixed across the four: model
  revision, rows, labels, the full specification with output wording adapted,
  non-thinking template, temperature zero. Costs are in the results table.
- Appendix on data sources and discrepancies. STRIDE reasoning counts (1,269, 1,230, 1,340
  of 1,350; 15.42% of 8,688) and the reason the inputs were not cleaned:
  `docs/findings/training_input_reasoning.md`. Validation set (3,012 rows,
  typed reasoning and `think` calls removed, visible text kept):
  `docs/decisions/cot_removed_id_validation.md`. OOD inputs per source (empty
  or name-only assistant bodies; no confirmed private reasoning):
  `docs/findings/ood_input_reasoning.md`. Inputs kept unchanged for
  comparability, and 6,247 rows that could not be restored:
  `docs/decisions/frozen_ood_comparability.md`,
  `docs/findings/ood_prose_restoration.md`. That decision notes that our
  frozen inputs are not proven to be the exact bytes of every run in the
  paper. Maximum training length 29,696 tokens: the 4B and 27B model cards.
- Appendix on prompts. Byte copies in `assets/prompts/`, SHA-256:
  `compact_instruction.txt` `39ab7657…` (from
  `experiments/tool_trajectory_monitoring/prompts/student.txt`),
  `full_specification.txt` `a5d403e6…` (from `.../prompts/teacher.txt`),
  `control_arena_binary.txt` `a9437d61…` (from
  `experiments/sleightbench_eval/prompts/control_arena_binary.txt`). Word
  counts 95, 701 and 299 are from `wc -w`. The ControlArena instruction was
  used in the exploratory SleightBench run `sb02`
  (`experiments/sleightbench_eval/README.md`); whether Section 6 reports that
  run is still to be decided.
- Hugging Face datasets named in the appendix on data sources and discrepancies, with full
  revisions: training set `adityaasinha28/deliberative-monitor-pipeline` at
  `2f10f4e49f4bf349965640ac021406689455d1c3`, file
  `merged/stride_gloom_crh_s45_o46_ctrl_t7_bal.parquet`
  (`docs/findings/tool_trajectory_teacher_cost_estimate.md`); OOD inputs from
  the same repository and revision under `filtered_eval/`
  (`docs/findings/ood_input_reasoning.md`); validation set from
  `aksh-n/rich-trajectories-cleaned` at `92a44c82…` (STRIDE `test`, 946 rows)
  and `Syghmon/gloom-data-exfiltration` at `ee3c56ba…` (2,066 rows after the
  length filter) (`docs/research/tool_transcript_monitoring.md`). The dataset
  cards declare no license, so we name the datasets and do not redistribute
  them. SleightBench's main setting removes all assistant text and thinking
  and keeps user turns, tool calls and tool results
  (`experiments/sleightbench_eval/README.md`, the `stripped` view).
- MLE-Sabotage inputs: one input was inspected on 2026-11-10
  (`data/tool_trajectory_monitoring/distillation_ood/prompts.jsonl`, row 956).
  It holds the user's task, then per step the agent's code under `Code:` and
  its run output under `Execution info:`; the audit in
  `docs/findings/ood_input_reasoning.md` records that visible `Plan:` text is
  kept and the separate thinking field is not. The per-benchmark counts of
  empty assistant bodies are in that finding and are no longer tabulated in
  the report.
- Section 3.2. Teacher cache (Kimi K3, full specification, temperature zero,
  reasoning off, 8,688 rows, normalization over the `0` and `1` logprobs):
  `docs/findings/tool_trajectory_training_teacher_cache.md`,
  `src/gleipnir/teachers/openrouter.py`. Loss (binary cross-entropy on the
  two-token margin, soft weight 1, hard-label weight 0, no temperature):
  `experiments/deception_distillation/train_student_sft.py`,
  `experiments/tool_trajectory_monitoring/distillation_config.yaml`. Filter
  (852 of 8,688 removed): `docs/decisions/strict_teacher_filtering.md`; its
  effect on one 4B seed (validation pAUROC@20 0.846812 to 0.895914, FPR at 0.5
  5.28% to 21.45%, no random-removal control):
  `docs/findings/teacher_disagreement_filtering.md`. Hard against soft, 4B
  (OOD 0.7458/0.7131/0.6902 against 0.7869/0.8125/0.7768; validation means
  0.8333 and 0.9026; BF16 serving): `experiments/hard_label_seeds/README.md`.
  Hard against soft, 27B (0.9044 and 0.9087 with the compact instruction and
  BF16; 0.9328 and 0.9508, validation 0.9587 and 0.9444, extreme scores 0.80
  and 0.20 in the release configuration):
  `experiments/qwen38_27b_hard_labels/README.md`. The text does not give the
  cost of the teacher cache: the recorded $224 is a lower bound on the spend.
- Table `tab:hard-soft`. 4B seed ranges and means, and the 27B rows:
  `experiments/hard_label_seeds/README.md` and
  `experiments/qwen38_27b_hard_labels/README.md`. Validation score of the
  hard-label 27B with the compact instruction: 0.966474
  (`results/qwen38_27b_hard_labels/comparison.json`), so 0.966. Extreme-score
  shares: 4B 0.82 and 0.01 (seed means), 27B 0.79 and 0.17 (BF16), 0.80 and
  0.20 (release configuration), all from the 27B experiment record.

BibTeX metadata for all entries comes from the arXiv API (read 2026-11-09).

- Section 3.3 (training setup). LoRA rank, targets, optimizer, schedule, 245
  updates: `experiments/tool_trajectory_monitoring/README.md` and the campaign
  configs. Training time: about one hour per 4B run (QAD run: 64 minutes,
  `experiments/qad_v02/README.md`), 4.5 hours for the 27B
  (`docs/findings/qwen38_27b_campaign.md`). No truncation: longest training row
  29,337 tokens against a limit of 29,696 (`experiments/qad_v02/README.md`).
  Seed average, release rule and 0.9124 against 0.9026:
  `experiments/seed_average/README.md` (seeds share initialization and differ
  in data order). QAD recipe (learning rate 1e-5, 245 updates, target is the
  averaged adapter's own BF16 score): `experiments/qad_v02/README.md`. 27B
  instruction choice, 0.9420 against 0.9216 on the validation set (BF16), chosen
  on the validation set, 9.9% more prompt tokens: `docs/decisions.md`
  (2026-10-20), `experiments/gleipnir27b_vocab_distillation/README.md`. LoRA
  citation (Hu et al., ICLR 2022, arXiv:2106.09685) entered from memory, not
  checked against the arXiv record.
- Appendix on the training configuration (`tab:training-config`). LoRA
  targets and dropout: `src/gleipnir/training/branch_model.py`. Warmup 3% and
  weight decay 0: `src/gleipnir/campaigns/monitoring/training.py`. Learning
  rate, updates, batch: campaign configs `data/teacher-seed-sdpa-s1-01/` and
  `data/qwen38-27b-s0-01/`. Rank 384 and alpha 768 of the averaged adapter:
  `experiments/seed_average/README.md`. QAD column: `experiments/qad_v02/README.md`.
  Not in the table: gradient clipping (the trainer default is a maximum norm
  of 1.0; not confirmed for these runs) and the 16,384-token packing budget.
- Appendix D.5 (instruction of the 27B). Validation 0.9216 and 0.9420, OOD
  0.9087 and 0.9352 (BF16), SHADE-Arena 0.588 and 0.730:
  `docs/decisions.md` (2026-10-20), `docs/findings/qwen38_27b_campaign.md`.
  The 9.9% token increase: `experiments/gleipnir27b_vocab_distillation/README.md`.
  Appendices D.2 to D.4 carry the text and numbers that were in Sections 3.1,
  3.2 and 4.2; their sources are unchanged.
- Appendix D.2, unadapted models against the trained monitors of Sinha et al.
  All six scores and costs are rows of Table `tab:ood-all`. The comparison has
  no intervals; the Qwen3.5-27B difference is 0.008.
- Table `tab:format` (Appendix D.2) repeats rows of Table `tab:ood-all`; no new numbers.
- Output instructions in the prompts appendix. Excerpts of the four prompt
  variants of the output-format ablation, cut from
  `experiments/tool_trajectory_monitoring/prompts/` (see `assets/README.md`);
  SHA-256 prefixes: `output_binary.txt` `a16700fd…`, `output_ordinal.txt`
  `e913d30d…`, `ordinal_score_bands.txt` `02fcfafb…`,
  `output_binary_reasoning.txt` `b7043aaa…`, `output_ordinal_reasoning.txt`
  `5e6dc59d…`. A diff of `teacher_ordinal.txt` against `teacher.txt` shows that
  the 0-10 variants also replace the decision boundary with five score bands
  and ask for a score that reflects confidence and severity; Appendix D.2
  (now C.2) was corrected to say so. The same is noted in
  `experiments/tool_trajectory_monitoring/QWEN_ORDINAL_OOD_BENCHMARK.md`.
- Appendix order since 2026-11-10: A setup of Sinha et al., B data, C results
  and ablations, D training configuration, E additional results, F prompts.
  Earlier notes that say "Appendix C" for prompts or "Appendix D" for results
  refer to the old order.
- Instruction lengths in tokens (prompts appendix and `tab:training-config`):
  133 (compact), 945 (full specification), 380 (ControlArena). Counted on
  2026-11-10 with `tokenizer.json` of `Qwen/Qwen3.5-4B` at revision `851bf6e`,
  on the text of each file in `assets/prompts/` without its final newline and
  without special tokens or the chat template (with the newline: 134, 946,
  381). "Both Gleipnir monitors use it": the Qwen3.8-27B tokenizer gives
  identical ids on the 3,012 validation prompts and the 7,836 training rows
  (`experiments/qwen38_27b_campaign/README.md`, token identity); not checked on
  these instruction files alone. Word counts (95, 701, 299) are no longer in
  the text.
- Appendix C.3 (teacher-agreement filter), additions of 2026-11-10. Updates
  272 to 245 (training tokens 83.8M to 75.9M) and "chosen for data efficiency
  and observed ranking": `docs/decisions/strict_teacher_filtering.md`,
  `docs/findings/teacher_disagreement_filtering.md`. Qwen3.5-9B data scaling
  (204 to 8,688 nested, source-label-balanced rows; rank-128 QLoRA, one epoch,
  learning rate 5e-5, batch 32, Kimi K3 soft targets, compact instruction; one
  seed; Mean-OOD 0.6988, 0.7076, 0.8111, 0.7907, 0.8365, 0.8227):
  `docs/research/tool_trajectory_distillation_ood_scaling.md`. "Quantized to 4
  bits" is NF4 QLoRA as in the project's other QLoRA runs; the scaling record
  says only "QLoRA". In Table `tab:filter` since the same day: the margin sweep, with all five arms
  rescored on validation and OOD through one BF16 serving path on 2026-11-10
  (validation 0.846, 0.896, 0.879, 0.837, 0.851 and Mean-OOD 0.776, 0.788,
  0.796, 0.767, 0.779 for unfiltered and margins 0, 0.05, 0.10, 0.20):
  `experiments/filter_margin_ood/README.md`. The margin was selected on
  validation (`docs/decisions/strict_teacher_filtering.md`); earlier records
  in `docs/findings/teacher_disagreement_ood.md`.
- Appendix C.5, 4B rows of `tab:instruction`. Validation 0.8773 (compact) and
  0.8762 (full specification): `experiments/gleipnir27b_teacher_4b/README.md`
  (seed 1, targets from Gleipnir 27B under the full specification, 8,239
  training rows after its own strict filter, each arm trained and scored with
  its instruction; one run per arm, no OOD scoring). No record was found of
  the released 4B scored with the full specification on the validation set or
  the OOD suite; `experiments/adversarial_is_it_4b/README.md` scores it that
  way only on an adversarial set.
- Appendix C.6 (weight averaging, `tab:average`). Per-run validation 0.8959,
  0.8978, 0.9142 (mean 0.9026) and Mean-OOD 0.7869, 0.8125, 0.7768 (mean
  0.7921); average 0.9124 / 0.8108; release rule and seed mean not reported (JF,
  2026-11-10: internal; C.6 says the OOD variation was seen before averaging): `experiments/seed_average/README.md`. Runs 1 to 3 are seeds 0
  to 2. The average was scored on a B300 and the runs on B200s (seed 1
  rescored on the B300: +0.0006); not stated in the report.
  `wortsman2022soups` was entered from memory (unverified), like `hu2021lora`.
- `xin2026qad` (Section 3.3, C.7) and `ouyang2025lowbit` (C.7): metadata and
  abstract claims checked against the arXiv pages and the ACL Anthology on
  2026-11-10 (abstract: "models with larger sizes or fewer training tokens
  experience less quantization-induced degradation"). Ouyang et al. means
  pretraining tokens; the report does not claim our 27B is undertrained.
- Appendix C.7 (serving precision, `tab:precision`). 4B before QAD (v0.2):
  BF16 0.9124 / 0.8108, optimized 0.8807 / 0.7579; released 4B (QAD): BF16
  0.9332 / 0.8039, optimized 0.9117 / 0.7703: `experiments/qad_v02/README.md`.
  27B with the full specification: BF16 0.9420 / 0.9352, optimized 0.9444 /
  0.9508: `experiments/optimized_gap/README.md` (which also has row-bootstrap
  intervals; not reported, at JF's request, 2026-11-10). The 27B BF16 scores use an FP32 LM
  head on BF16 hidden states (`gleipnir27b_vocab_distillation/extract01`).
- Appendix C.5, 4B rows of `tab:instruction` (replaces the 27B-teacher pair).
  Compact 0.9332 / 0.8039: `results/qad-v02-bf16-01/evaluation/`. Full
  specification 0.9247 / 0.8032: `experiments/gleipnir4b_full_specification/README.md`
  (run of 2026-11-10, `results/gleipnir4b_full_specification/comparison.json`).
- Section 4 (planned serving table). 27B ladder (full320, c128, EU-RO-1 B200,
  2026-10-18): BF16 23.1K tokens/s; last-query and lean GDN at BF16 +2.3%;
  FP8/FP4 projections 2.07×; MXFP8 attention 2.14×; fused producers and tuned
  tiles 2.29×: `experiments/qwen38_27b_serving/README.md`. 4B ladder (same
  workload and harness, US-CA-2 B200, 2026-11-10): BF16 118.9K; low precision
  1.77×; last-query 1.011×; route gate 1.002×; lean host path 1.059×; total
  1.90× (226.0K): `experiments/gleipnir4b_serving_ladder/README.md`. Hosted
  prices and B200 rates: `docs/research/tool_trajectory_inference_economics.md`.
  Table `tab:serving` (drafted 2026-11-10). Hosted list price for the
  released prompts: 4B $0.2598 (compact, 55,346,879 OOD prompt tokens), 27B
  $1.8476 (full specification, 60,539,619 tokens), and final costs $0.0850 /
  $0.0366 and $0.3896 / $0.1677 from measured OOD wall time (244.9 s and
  1,122.6 s): `docs/research/tool_trajectory_ood_frontier.md`. Middle rows are
  the final cost times the gains. 27B gains: total 2.28× (released `graphs01`),
  unneeded work 1.0228× (measured at BF16, before quantization), low precision
  2.28 / 1.0228 = 2.23×. The 4B benchmark projection reproduces $0.0850.
  Open: the full specification adds 9.4% prompt tokens on the OOD suite by
  these counts, while Section 3.3 and C.5 say 9.9%
  (`experiments/gleipnir27b_vocab_distillation/README.md`); basis to check.
  Since 2026-11-10 (JF) the table is in USD per million prompt tokens: API
  input prices $0.03 and $0.195 (the one output token is ignored); final
  $0.0098 / $0.0042 (4B) and $0.0412 / $0.0177 (27B) are the per-1,000 costs
  divided by 8,654.7 and 9,466.7 tokens per evaluation.
  Since later on 2026-11-10 (JF: one workload for every row) all self-hosted
  cells are the hourly rate divided by the benchmark median throughput: 4B
  118,933 / 210,609 / 225,964 tokens/s; 27B 23,093 (control02) / 51,881 /
  53,064 (`graphs01`), where 51,881 is 53,064 divided by the 1.0228 measured
  at BF16. The 27B total is 2.30× by these medians (the 27B record rounds to
  2.28×). OOD-measured per-token costs: 4B $0.00982 (identical), 27B $0.0412
  (benchmark $0.0418, 1.6% higher).
- Section 4.1 (self-hosting, drafted 2026-11-10). Classification mode: vLLM
  0.31 pooling runner with `--convert classify`, LAST pooling, raw logits of
  the `0`/`1` tokens (`README.md`, `experiments/release_serving/stock_vllm.py`,
  which says the recipe runs on any GPU). The benchmark's BF16 rows used the
  project's BF16 launcher (`Bf16Worker`, score endpoint on the same pooling
  runner), not a literal `vllm serve`; the literal recipe on the same B200
  class gave 106K tokens/s on the validation set for the 4B
  (`experiments/gleipnir4b_full_specification/README.md`). Benchmark: 320
  trajectories, concurrency 128, prefix caching off
  (`experiments/b200_inference_benchmark/README.md`). Rates $7.99 (Runpod on
  demand) and $3.44 (CoreWeave $8.60 with "up to 60%" committed discount):
  `docs/research/tool_trajectory_inference_economics.md`.
  Revised the same day (JF): throughput numbers moved out of the text to a
  planned appendix table (`app:serving-throughput`, stub); added the
  prefill/decoding explanation (prefill compute-bound, decoding
  memory-bound; general background, no citation yet). Output-to-input price
  ratios 5 (4B: $0.15 / $0.03) and 8 (27B: $1.56 / $0.195):
  `docs/research/tool_trajectory_inference_economics.md`.
- Section 4.2 (low-precision arithmetic, drafted 2026-11-10). Recipe: FP8
  (E4M3, per-channel weight scales, dynamic per-token activation scales) on
  the QKV/output projections of the full-attention blocks, FP4 on the MLP and
  GDN projections, MXFP8 attention; BF16 cache/recurrence and two-row
  classifier, FP32 gates/state; small linears stay BF16:
  `docs/decisions/b200_monitor_score_reference.md`,
  `docs/findings/b200_attention_precision.md`. The text does not name the GDN
  layers ("the other large projections"). Score changes: Table
  `tab:precision`. B200 only: the kernel receipts are hardware-bound and the
  server refused a B300 on 2026-10-20 (`experiments/optimized_gap/README.md`).
  Repeatability: two passes on one server, 4B per-row MAE 0.0110 (validation)
  and 0.0090 (OOD), 27B 0.0065 and 0.0057; |Δ pAUROC@20| ≤ 0.0013 on every
  split: `experiments/release_serving/README.md`.


- Section 4.1, Decider price comparison: OpenRouter list price of the
  `pplx-decider-v1.2` endpoint (`-20261016`), $0.02/M input tokens, no output
  charge, from `docs/findings/pplx_decider_ood_benchmark.md` (Cost) and
  `docs/research/tool_trajectory_inference_economics.md`. Hosted serving
  precision and hardware are not observed. Compared with the 27B committed
  cells of Table `tab:serving` (0.0414 BF16, 0.0180 final).

- Appendix E.2 (low-precision kernels), collected 2026-11-14 from repo records:
  recipe (16 FP8 W8A8 attention projections with per-channel weight and
  per-token activation scales on native CUTLASS GEMM; 64 FP4 MLP and 48 FP4
  GDN projections on FROST; cuDNN MXFP8 prefill) from
  `docs/decisions/b200_monitor_score_reference.md`; FROST is
  `cudnn.gemm.frost` (`src/gleipnir/serving/frost_wrappers.py`), tile tuning
  from `docs/findings/b200_fp4_gemm_tuning.md`, fused activation packing from
  `docs/findings/b200_fp4_serving_preparation.md`; MXFP8 forward kernel
  `SdpaFwdDslSm100` D256, exp2 replacement and paged-cache bridge from
  `docs/findings/b200_mxfp8_serving.md`; backward test (18.71% adapter-gradient
  difference against a 10% ceiling, 29.64% slower update) from
  `docs/findings/b200_nvidia_mxfp8.md`; QAD arithmetic (serving MXFP8 forward,
  quantized-QKV FA4 surrogate backward, straight-through projections) from
  `experiments/qad_v02/README.md`; 27B port from
  `docs/research/qwen38_27b_serving_transfer.md`. The two NVIDIA bib entries
  carry no year or version; library versions of the release image are not
  yet recorded in the report. The 8/24 layer split of the 4B is from
  `docs/findings/b200_nvidia_mxfp8.md`.

- Appendix E.2, fused SwiGLU kernel: adapted from NVIDIA's SM100 block-scaled
  dense SwiGLU template (`docs/findings/b200_fp4_swiglu_fusion.md`); direct
  FP4 output selected for the 4B on 2026-10-07
  (`docs/decisions/b200_native_fp4_output_inference_baseline.md`, +1.88%
  c128 throughput; used above 1,536 rows, original path below) and listed in
  the current recipe (`docs/decisions/b200_monitor_score_reference.md`);
  ported to the 27B and rejected, no end-to-end gain
  (`docs/research/qwen38_27b_serving_transfer.md`).

- Section 4.3 (removing unneeded work), 2026-11-14. Per-change gains for the
  4B from the matched ladder (`experiments/gleipnir4b_serving_ladder`,
  session 2 medians): last-query 1.011x, route 1.002x, lean GDN 1.059x. For
  the 27B at BF16 (`experiments/qwen38_27b_serving/README.md`, candidates
  table): last-query +0.40%, fast GDN +1.99%, both +2.28%. Layer counts 32
  (8/24) and 64 (16/48), final layer full attention (layers 31 and 63):
  `docs/research/qwen38_27b_serving_transfer.md`. Bitwise-unchanged scores at
  concurrency 1 and the 10.3% lower latency at 1M tokens (4B, 43.08 s to
  38.64 s): `docs/findings/b200_last_layer_trim.md`. Lean GDN host path
  (boundaries once per forward, in-place output, zero-state start, bitwise
  equal): `experiments/b200_gdn_fast/README.md`. Route gate (context-parallel
  variant slower under 8,192 tokens, 2.3-2.9 ms on short single requests,
  canary MAE 0.0005, 4B only): `docs/findings/b200_gdn_route.md`,
  `docs/decisions.md` (2026-10-15). The route gate changes arithmetic for
  short requests, so the bit-for-bit statement covers only the two main
  changes. "One full-attention layer follows every three GDN layers" is
  inferred from the 4B attention layer indices 3, 7, ..., 31 and the 16/48
  split of the 27B.
- Appendix E.1 (throughput and latency, `tab:serving-steps`,
  `fig:serving-latency`), measured 2026-11-15 on Pod `nsw57y1tcxh9gw`
  (EU-RO-1): `experiments/serving_concurrency/README.md` (Result),
  `results/serving_concurrency/summary.json` (copied to
  `assets/serving_latency.json`) and `environment.json`. Figure:
  `python -m experiments.serving_concurrency.plot --output
  assets/serving_latency.pdf` (and `.svg`). The 28 virtual cores and 283 GB
  are the Pod's allotment (`RUNPOD_CPU_COUNT`, `RUNPOD_MEM_GB`); the host CPU
  has 224 threads. Second session the same day on the same machine and GPU
  (Pod `179xu9pmh64fxh`; README sections "Per-change breakdown and 4B graph
  sweep" and "Diagnosis of the score change",
  `results/serving_concurrency/breakdown01/summary.json`). Rows of
  `tab:serving-steps`: 4B stock and low precision from the first session
  (`4b_bf16`, `4b_quant`); 4B graphs, last-token and kernel-variant rows
  from `4b_quant_g2048`, `4b_lastquery_g2048`, `4b_route_g2048` (graph
  sizes in multiples of 32); 4B last row from
  `breakdown01/sweep_e/summary.json` (`4b_fast_g2048e`, the layout with
  bit-identical scores; its gain is 227,709 / 214,885). All 27B rows from
  the second session (`27b_bf16`, `27b_quant_nograph`, `27b_quant`,
  `27b_lastquery`, `27b_fast`). The "at most 2%" between sessions is the
  largest difference of a repeated figure (27B released p50 at one request,
  23.0 against 22.5 ms). Figure: the 4B released curve is `sweep_e`; the
  other curves and the whole right panel are from the first session
  (prompts from 4,096 tokens do not use the larger graphs). Concurrency 16
  against 128: 4B 225,109 / 227,709 tokens/s, p50 282.7 against 2,158 ms;
  27B 52,843 / 52,864, p50 1,199.6 against 9,553 ms; 4B peak 229,031 at
  32. Bit-for-bit claim: `fast_g2048e` against `fast` at concurrency 1 on
  3,012 ID, 6,395 OOD and 320 benchmark rows, 0 scores changed. Stock vLLM
  with the same graphs: `4b_bf16_g2048` 15.0 ms, `27b_bf16_g2048` 36.7 ms.
  Table 1 was not changed: its 27B gains (2.21x, 1.04x) are from the first
  session, its 4B gains (1.77x, 1.07x) from the US-CA-2 ladder; here they
  are 2.22x and 1.04x, and 1.76x and 1.09x (227,709 / 209,757). The released
  image does not yet record the larger 4B graphs (JF, 2026-11-15: the
  deployment stack will be updated).
  Tokenizer (same Pod, README section "Tokenizer"): the "+ Gigatoken
  tokenizer" rows, the released curves of both panels of the figure, the
  concurrency figures (c16 281 and 1,197 ms; peaks 229,608 and 53,119) and
  the prompt-length figures (2.94 and 10.75 s; 36.3 and 125.9 s) are from
  `results/serving_concurrency/giga01/summary.json`. "About 5 ms" is the
  server's count inside Transformers (9.1 s over 1,984 prompts for the 4B,
  9.8 s for the 27B); 15 ms and 0.1 ms are the direct timing and the
  server's Gigatoken count (0.19 s over 2,304 prompts). "Same tokens" was
  checked on the 320 benchmark prompts, and scores at one request at a time
  were identical in the matched comparison (`tokenizer02`). Until this
  measurement the vLLM 0.31 stack tokenized with Transformers (the Gigatoken
  patch was inert), so all other rows and the stock curves use Transformers.
  The released image does not contain a working Gigatoken hook yet.
  Benchmark prompt lengths (320 rows, 1,310,581 tokens, the same prompts for
  both models): mean 4,096, median 503.5, quartiles 311 and 4,499, 95th
  percentile 21,564, range 188 to 28,733; from `prompt_tokens` in
  `results/serving_concurrency/4b_fast/full_c1_repeat0.json`.
