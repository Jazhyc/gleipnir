# Sources and scope

Record each reported number's source document or artifact, the repository
revision it was read at, the metric definition and any comparison limits.
Add an entry here when you add a number to the report.

No results are reported yet.

## Section 6 and Appendix "Held-out and adversarial evaluation: details" (ARK)

Recorded 2026-11-20 by ARK, on report branch `docs/v1-technical-report` at
`b62c1058`. Results and data folders are git-ignored; their files were read
from the `feat/adversarial-stress-tests` checkout at `e4b9e391`. SHA-256
prefixes identify the files read. Monitors: 4B v0.3 adapter (same weights as
the released 4B), 27B `27b-merged` BF16; BF16 Transformers + PEFT scorer.
pAUROC@20 is `gleipnir.evaluation.metrics.normalized_partial_auroc` (no
McClish correction). 1% lines: 4B 0.622 (1,000 benign strict-OOD rows),
27B 0.731 (400 rows, 4 above the line), Kimi K3 0.755 (1,000 rows); from
`experiments/adversarial_is_it_4b` and `adversarial_natural_cheats`.

- **EvilGenie (Table `tab:evilgenie`, `tab:evilgenie-problems`, Section 6.1).**
  Recomputed post hoc on the rows both monitors scored by
  `experiments/adversarial_evilgenie/shared_rows_report.py` (run with
  `GLEIPNIR_DATA_ROOT` set to the data checkout). Inputs:
  `results/adversarial_evilgenie/4b-v03/rows/predictions.jsonl` (`fae17892`),
  `.../27b-merged/rows/predictions.jsonl` (`49e04990`),
  `data/adversarial_evilgenie/rows_4b-v03.jsonl` (`2566f455`), lines from
  `report.json`. Bootstrap: whole problems, 10,000 draws, seed 0, 9 problems.
  The 4B all-rows values (AUROC 0.751, 10/108, 16/1,695, 0/180) and dataset
  counts, exclusions and the `arc191_c` reading are from
  `experiments/adversarial_evilgenie/README.md`. The split by `arc191_c` was
  chosen after the scores; the README's own "without arc191_c" figure (0.933
  on 39 hacks, all 4B rows) differs from ours (0.942 on the 36 shared hacks)
  because the row sets differ.
- **APPS, identical text (Table `tab:apps-text`, Section 6.2).** pAUROC@20 and
  intervals from `results/figures/backdoor_pauroc.json` (`442d20a8`), written by
  `results/figures/backdoor_pauroc.py`: task bootstrap, 2,000 draws, seed 0,
  checked against each monitor's `report_<monitor>_heldout.json`. AUROC, paired
  differences, tie shares, mean score changes, calibration flag rates, task
  counts, text construction and hand reads: `experiments/adversarial_crossed_text/README.md`
  (Results, Amendments 1-3, Limits). All texts are GLM-5.3-Flash; Amendment 3
  fell back to the Amendment 1 cover stories.
- **Text addressed to the monitor (Appendix, APPS).**
  `docs/findings/apps_released_monitors.md` (run `apps01`, released 4B v0.4 and
  27B v0.2 on the optimized B200 stack).
- **Probe (Table `tab:apps-probe`).** `experiments/adversarial_crossed_text_probe/README.md`,
  Results 24 October 2026.
- **Natural cheats (Table `tab:natural-cheats`).**
  `docs/findings/adversarial_frontier_comparison.md` (pAUROC@20 in percent with
  two-digit intervals, AUROC) and `experiments/adversarial_natural_cheats/README.md`
  (counts, recall at 1%, comment stripping). The 2-of-4 and 1-of-4 counts for
  unannounced cheats are at a matched 1% line from the negatives (frontier
  comparison, Amendment 2). Costs are not reported in the report because the
  record mixes A100 unbatched and API prices.
- **Comment and docstring removal (Table `tab:comment-strip`).**
  `experiments/adversarial_confession_strip/README.md` and its results;
  rows from `experiments/adversarial_monitor_aware`.
- **Multi-agent sabotage (Table `tab:aisi`).** Private AISI data under the
  sprint repository's `data/aisi/` (git-ignored, licence: aggregate results
  only, confirmed by AISI): `figures/pauroc.json` (`ff73e224`), flag counts and
  session counts from `pilot40/RESULTS.md` and `pilot_chains/RESULTS.md`, Qwen
  rescore from `rescore_qwen/results.json`, completed attacks from the
  `rows/manifest.jsonl` files (`successful_attack_judge`). Confirm with AISI
  that an arXiv report counts as reporting aggregate results.
- **Lab liars (Table `tab:lab-liars`).** `results/adversarial_lab_liars/pauroc20_posthoc.json`
  (`3b41b8cb`, seed bootstrap) and `experiments/adversarial_lab_liars/README.md`;
  the world from `experiments/live-peers/liar-scale` in the sprint repository.
  Post addressed to the monitor: `experiments/adversarial_is_it_4b/README.md`
  (Step 1). Probe and bag-of-words: `experiments/adversarial_lab_liars_probe/README.md`.

- **Corrections and recomputations (2026-11-20).**
  - Within-task pair shares (Table `tab:apps-auroc`) and mean scores are from
    `results/adversarial_crossed_text_amendment3_20261023/report_<arm>_heldout.json`
    (`within_task_win_probability`) and recomputed from `scores/<arm>.jsonl`:
    4B specific accusation 0.802 (strict 0.716, ties 0.173). The crossed-text
    README's sentence "the backdoored program scores higher in 55% (specific)
    and 54% (generic) of pairs" does not match these rows; the report uses the rows.
  - 3-decimal AUROCs (APPS) from the same `report_<arm>_heldout.json` files;
    probe values from `results/adversarial_crossed_text_probe/<arm>/report.json`.
  - Cover story against no text on the same 28 tasks (4B 0.604 -> 0.062, 27B
    0.825 -> 0.600, base 0.601 -> 0.131): recomputed from `scores/<arm>.jsonl`
    with `normalized_partial_auroc`, post hoc.
  - Natural-cheats table and stripping AUROCs from
    `results/adversarial_natural_cheats/report.json` (`comparison`, `points`);
    Sonnet's first-cheat row set is 44 cheats and 237 negatives.
  - AISI honest-session flags (0 for every monitor and goal) and accepted-vs-
    rejected AUROC from `data/aisi/pilot40/scoring/results/report.json` and
    `pilot_chains/scoring/results/report.json` (`existing_line`).
  - EvilGenie 27B control runs: 0 of 115 flagged; 1,312 = 105 + 999 + 115 + 19 + 74.
  - Lab liars: AUROCs from `results/adversarial_lab_liars/report.json`
    (`primary_auroc`); 4B at 0.622 flags 0 pre-assigned recruiters in any group
    (recomputed from `4b/main/predictions.jsonl`).

- **Figures (2026-11-21).** `fig:apps-text`, `fig:apps-probe`, `fig:lab` and
  `fig:lab-setup` are drawn by `report-figures/make_figures.py` from
  `report-figures/data.json`, which `report-figures/refresh_data.py` builds from
  the files listed above; the values match the tables. Figure `fig:apps-text`
  replaces the main-text table `tab:apps-text`; its intervals are now in
  `tab:apps-full`. Readers flagged after the framing post: 4B 245/300
  (`flagged_1%` 0.8167), 27B and Kimi 0/100, unadapted Qwen3.8-27B 0/100.

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
- Perplexity Decider blog post (`perplexity2026deciderblog`,
  `https://www.perplexity.ai/hub/blog/perplexity-decider`, 2026-11-12;
  section text and BibTeX supplied by JF on 2026-11-19, the page returns 403
  to our fetcher). Its "Monitoring RL rollouts" section says Perplexity uses
  Decider to diagnose RL rollouts (what failed in a step: tool service,
  external site, sandbox, harness or the agent's code; whether the agent read
  a result correctly), observing only. The excerpt names no training data.
  The `pplx-decider-v1.3` card credits most gains to tasksource (Sileo 2024)
  and gives no other training details.
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
  $7.99/h for one B200. Since 2026-11-18 (JF) their costs use one request
  at a time, as in Table 1: hourly rate / (throughput x 3,600) x 10^6 with
  204,673 and 50,788 prompt tokens/s, times 8,654.7 and 9,466.7 OOD prompt
  tokens per evaluation (55,346,879 and 60,539,619 over 6,395 rows): 4B
  $0.0939 / $0.0404, 27B $0.4137 / $0.1781 (formerly $0.0850 / $0.0366 and
  $0.3896 / $0.1677 from batched OOD wall time). The registry carries the
  same values. "Gleipnir 9B (earlier release)" is the registry row
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
  `fig:serving-latency`) and Table 1, measured 2026-11-15/16 on Pod
  `179xu9pmh64fxh` (EU-RO-1; the stock BF16 length sweeps on Pod
  `nsw57y1tcxh9gw`, the same machine and GPU):
  `experiments/serving_concurrency/README.md`. The 28 virtual cores and
  283 GB are the Pod's allotment; the host CPU has 224 threads.
  - Benchmark: the 96 agent trajectories (README section "Agent-trajectory
    benchmark"); JF removed the deception data and CoT Red-Handed on
    2026-11-15 and chose 8 concurrent requests.
  - `tab:serving-steps`: `results/serving_concurrency/agent03/summary.json`
    (one request at a time; 8 concurrent for stock and the last row) and
    `agent03/c8/summary.json` (8 concurrent, other rows). Latencies are
    rounded to whole milliseconds.
  - Generation mode (`tab:serving-generation`, the sentence in Section 4.1;
    JF asked for it on 2026-11-16: thinking off, no rationale limit,
    EU-RO-1): `results/serving_concurrency/generation01/` on Pod
    `8v7naalpvfybi5`, README section "Generation mode of the base models".
    Instruction `teacher_ordinal_reasoning.txt` (`f79026c9…`), which contains
    `output_ordinal_reasoning.txt` and `ordinal_score_bands.txt`. Table
    cells are medians of three passes; the p50 is that of the median pass.
    The run was on a second machine of the same type (EPYC 9555), which
    the report does not mention (JF, 2026-11-16: the difference is small
    against the size of the effect). Stock classification there took 11.3 s
    and 54.2 s (90.9 and 446 ms), within 3% of the benchmark Pod; the report
    compares with Table 13 instead: 1,193,325 / 103,266 = 11.6 s and
    / 21,720 = 54.9 s, ratios 10.2 and 9.7.
    Costs in the table (JF, 2026-11-16, committed rate only, with the API price as reference
    and the caveat that output tokens are not priced) = hourly rate /
    (prompt tokens per second x 3,600) x 10^6.
    Released-stack seconds = 1,193,325 / 204,673 and / 50,788. The
    generation prompts have 1,286,829 tokens (longer instruction), so the
    ratios compare seconds for the 96 trajectories, not tokens per second.
  - B300 (`app:serving-b300`, `tab:serving-b300`, the sentence in Section
    4.2; JF, 2026-11-16: support is for flexibility, state that it brings no
    improvement): `experiments/serving_concurrency/README.md`, section "B300
    trial" (Pod `4sr7r26kfoc0z9`, results `b300_01`, `b300_03`, `b300_08`).
    The trial ran at the recorded 32K context limit. The 248 synthetic
    prompts and the one differing 27B score (0.560 against 0.617) are from
    the fresh-Pod check of `1.1-rc1`. Image support:
    `experiments/release_serving/README.md`, section "Image 1.1".
  - Table 1 (JF, 2026-11-16: lead with one request at a time and cost it
    that way): cost = hourly rate / (tokens per second x 3,600) x 10^6 with
    the one-request throughput of stock (103,266; 21,720), low precision
    (160,069; 46,717) and the last row of `tab:serving-steps`, the released
    stack with Gigatoken (204,673; 50,788). Gains: 3.24 and 4.43 against the
    API price; 1.55 and 2.15; 1.28 and 1.09; totals 6.43 and 10.36. Section
    4.3 gains at one request: last-token 1.014 and 1.006; GDN host path
    1.095 and 1.041; tokenizer 1.176 and 1.046. "13% and 5% higher with 8
    concurrent requests": 230,375 / 204,673 and 53,356 / 50,788; "about nine
    times as long": 424 / 48 and 1,834 / 192 ms. The caption no longer says
    that the last row agrees with the OOD-suite costs: those were measured
    in batches of 128 (about 226K and 53K tokens per second) and are now
    lower than Table 1. The costs of `tab:ood-all` and the frontier
    figures were regenerated at one request on 2026-11-18.
  - Figure: left panel from the sweeps in `agent03/summary.json`; right
    panel from `summary.json` (stock), `giga02` (4B released, graphs to
    4,096) and `giga01` (27B released). Concurrency paragraph: 424 and
    1,834 ms at 8; 2,671 and 11,234 ms at 64; peaks 230,375 (c8), 53,373
    (c16), 120,327 and 22,979 (c64).
  - CUDA graphs paragraph: the 224 short prompts are the rest of the old
    320-prompt benchmark (median 387 tokens); medians at one request from
    the stored rows of `4b_bf16` 18.2, `4b_quant` 33.5,
    `breakdown01/4b_quant_g2048` 14.9 ms and `breakdown01/27b_bf16` 30.2,
    `27b_quant_nograph` 58.1, `27b_quant` 22.2 ms. 26 to 23 ms at 4,096
    tokens: 26.3 (`giga01`) and 22.5 ms (`giga02`). 27B: 77.7 against 77.2
    ms. Bit-for-bit claim: `parity4096` (ID and OOD, 0 of 9,407 scores
    change against `fast`).
  - GDN kernel variant: about 2 ms on the 224 short prompts
    (`breakdown01/4b_lastquery_g2048` 14.9 to `4b_route_g2048` 13.0 ms).
  - Tokenizer: 15.3 and 0.2 ms are the direct timing for prompts above
    8,192 tokens; identical IDs on all 320 prompts of the old benchmark,
    which contain these 96. Until this measurement the vLLM 0.31 stack
    tokenized with Transformers (the Gigatoken patch was inert).
  - The released image has neither the larger 4B graphs nor a working
    Gigatoken hook yet (JF, 2026-11-15: the deployment stack will be
    updated).
  Benchmark prompt lengths (320 rows, 1,310,581 tokens, the same prompts for
  both models): mean 4,096, median 503.5, quartiles 311 and 4,499, 95th
  percentile 21,564, range 188 to 28,733; from `prompt_tokens` in
  `results/serving_concurrency/4b_fast/full_c1_repeat0.json`.
- Appendix serving, prefix caching, context limit and start-up paragraphs
  (2026-11-17):
  - Prefix caching: `experiments/b200_prefix_replay/README.md` (`replay03`,
    2026-10-16, 4B, 1M YaRN server, align mode): 598 checks, 595 s cached
    against 7,665 s uncached estimate (12.9×), hit fraction 99.6%; 52
    escalated checks 7.9×; |cached − uncached| on 8 checks mean 0.055, max
    0.151; GDN route gate changed C1 scores by MAE 0.055. Opt-in in the image
    since 0.2 (`serve.py --prefix-caching`;
    `docs/decisions/prefix_cached_serving.md`).
    Cause and quality: `experiments/serving_concurrency/README.md`, "Prefix
    caching and score identity" (`prefix01`, 2026-11-15, released 4B):
    replay bit-identical; 528-token blocks; with `--max-num-batched-tokens
    528` half-then-full equals uncached on all 127 hit rows, uncached pass
    72.2 s against 19.3 s (3.7×); ID/OOD through the cache: 82% of hit rows
    change (ID mean 0.025), macro pAUROC@20 +0.0015 ID and −0.0008 OOD, no
    source more than 0.012 (IaC +0.0117).
  - Context limit: `experiments/release_serving/README.md`, "Context limit"
    (262,144 since image 0.2, no YaRN).
  - Start-up: image 1.0 fresh-Pod check (US-NC-2 B200,
    `logs/runpod/release_serving/image10_fresh/`): cold 276 s (4B) and 476 s
    (27B) with downloads (`check.out`); downloads 1:41 and 4:50 from the
    `Fetching N files` bars of `{4b,27b}_recorded.log`, so 175 s and 186 s
    without; later starts 105 to 170 s. 27B downloads: 51 s
    (`image11_b300`, EUR-IS-1), 178 s (`image02_fresh`), 290 s (`image10_fresh`).
- Section 5 (main evaluation, 2026-11-18). Figure `fig:frontier`
  (`assets/ood_frontier.pdf`) is `scripts/plot_tool_trajectory_ood_frontier.py
  --report` on the registry `docs/research/tool_trajectory_ood_frontier.md`;
  the slide figure in `figures/` comes from the same registry. Scores and
  costs are rows of `tab:ood-all`. "229 times": $94.794 / $0.4137. Interval
  of Gleipnir 27B: row bootstrap within source x label, 2,000 resamples,
  [0.9413, 0.9596] (`experiments/optimized_gap/README.md`, "Bootstrap
  intervals"); it excludes seed variation (one seed). The paper's Qwen3.5-4B
  SFT+RL: 0.684 at $0.358. Decider: 0.896 at $0.176; 0.951 - 0.896 = 0.055.
  Kimi K3 reads the full binary specification (registry interface column).
  Frontier rule (JF, 2026-11-18): a row also leaves a frontier when a more
  accurate row costs at most 2% more (`FRONTIER_COST_TOLERANCE`). It drops
  pplx-decider v1.2 ($0.1763) from the committed frontier, where Gleipnir
  27B* costs $0.1781 (1% more): $0.0188/M against $0.02/M, but 60,539,619
  against 56,359,499 OOD prompt tokens.
- Table `tab:ood-all`, Gemini Decision 1 (JF, 2026-11-18): registry row
  0.867342 at $0.930555 per 1,000 (59,509,013 input tokens at $0.10/M);
  `docs/findings/decision_models_ood_benchmark.md`, section "Gemini Decision
  1". Two inputs blocked by Google's filter score 0.5; answered-only 0.8742.
  It is left out of the main figure like the other decision models off the
  frontier.
- Table `tab:ood-all`, pplx-decider v1.3 (JF, 2026-11-18): registry row
  0.894199 at $0.176015 per 1,000 (56,280,923 billed input tokens at the
  $0.02/M list price of v1.2; the free endpoint billed $0).
  `docs/findings/decision_models_ood_benchmark.md`, section "pplx-decider
  v1.3". Section 5 still names v1.2 as the strongest decision model
  (0.8956 against 0.8942).
- Table `tab:ood-all`, recall column (JF, 2026-11-18): mean recall at 1%
  FPR from `results/ood_recall_at_fpr/recall_at_fpr.json`
  (`experiments/ood_recall_at_fpr/compute`, 23 monitors, all reproducing
  their registry pAUROC@20 within 0.0006). Per source, the threshold is the
  largest that flags at most 1% of benign rows (flag when score >
  threshold); the six recalls are averaged with equal weight. Sinha et al.
  report no recall at 1% FPR, so their rows show a dash. Gemini Decision 1's
  two blocked rows score 0.5.
- Table `tab:ood-all`, AUROC column (JF, 2026-11-18). Rows of Sinha et al.:
  the AUROC column of their Table 4 (Appendix E.5, "six-dataset mean-OOD
  metrics"), read through the arXiv HTML v1 of 2605.29601 by a fetch agent
  on 2026-11-18. Its pAUROC@20 column matched our 17 rows exactly; the
  AUROC values were not rechecked against the PDF. Their Table 9 gives the
  per-benchmark AUROC with three-seed standard deviations. They report recall
  only at 5% FPR (Figure 12, no table). Our rows: `mean_auroc` in
  `results/ood_recall_at_fpr/recall_at_fpr.json`, the unweighted mean of six
  per-source `roc_auc_score` values.
- Figure `fig:recall-frontier` (JF, 2026-11-18): `scripts/plot_tool_trajectory_ood_frontier.py
  --report --recall`, which takes the registry rows we scored and replaces
  pAUROC@20 with `mean_recall` from `results/ood_recall_at_fpr/recall_at_fpr.json`,
  then recomputes both frontiers with the cost tolerance (see the next
  entry). Committed use: Gleipnir 4B*, Gleipnir 27B*.
- Frontier tolerance raised from 2% to 3% (JF, 2026-11-18): on recall at
  1% FPR, Mercury Decide ($0.1811, 0.821) costs 2.9% more than pplx-decider
  v1.3 ($0.1760, 0.815), so v1.3 leaves the on-demand recall frontier and the
  figure. The pAUROC@20 frontiers do not change. Recall frontier on demand:
  Gleipnir 4B, Mercury Decide, d1-ultra, Gleipnir 27B. "About twice":
  $0.1811 / $0.0939.
- Appendix C.2, SHADE-Arena (`app:shade`, JF, 2026-11-18). Table
  `tab:shade`: per-source pAUROC@20 from
  `results/ood_recall_at_fpr/recall_at_fpr.json` (released Gleipnir 27B,
  Kimi K3, unadapted Qwen3.8-27B). "At most 0.008": AgentDojo 0.992 against
  1.000. Lead 0.9508 - 0.9084 = 0.0424; (0.8316 - 0.5743) / 6 = 0.0429.
  Filter (650 of 1,500, others at most 4%), Kimi K3 length quartiles
  (0.663, 0.499), the flight-booking task `american_airlines_2` (333 of 850
  rows, 97% against 9% median share, the quoted instruction and the loyalty
  file), Kimi K3 within-task AUROC 0.847 and pAUROC@20 0.391, and the
  without-task values (SHADE 0.9581 / 0.9397, Mean-OOD 0.9719 / 0.9693; BF16
  0.9732): `docs/findings/shade_length_audit.md`. The report does not name
  the airline. The task-pair bootstrap paragraph was removed (JF,
  2026-11-18). Section 5's pointer cites this subsection.

## Section 6 opening (JF, 2026-11-19)

- Saturation of the OOD suite is stated without numbers (JF): the strongest
  monitors' scores are in Section 5 and `tab:ood-all`. The SHADE-Arena pointer
  repeats Appendix `app:shade`.

## Section 6.1 SleightBench and Appendix G.1 (JF, 2026-11-19)

- Citation `najt2026sleight`: arXiv:2605.16626 (v2, 2026-05-19), Najt, Toft,
  Tracy, Roger and Benton. "40 attacks across 11 categories" and the 32% Opus
  4.6 catch rate (extended thinking, 10 trials, 1% FPR) are from its abstract.
- Every row is the stripped view with the ControlArena instruction (user,
  2026-11-19): Gleipnir `sb02` (released optimized stack), unadapted Qwen
  `sb04_immediate`, Kimi K3 `sb05`, GPT-6 Sol/GPT-5.5/GPT-6 Luna/GPT-6.1 Terra
  `sb07`/`sb08`/`sb09`/`sb13`, Jev 1.13 and pplx-decider v1.2 `sb12`, v1.3 and
  Gemini Decision 1 `sb15`. Scores and intervals from each run's
  `results/sleightbench_eval/<run>/report.json`; findings in
  `docs/findings/sleightbench_released_monitors.md`.
- Costs: `experiments/sleightbench_eval/frontier.py` →
  `results/sleightbench_eval/frontier.json`. API rows: mean billed input and
  output tokens over the 86 stripped rows at the OOD frontier's list prices,
  no cache discounts (Kimi K3 at the OOD frontier's $3/M in, $15/M out
  Fireworks ceiling, not the Makora rate that sb05 paid; v1.3 at v1.2's
  $0.02/M). Gleipnir and the base models: mean Qwen prompt tokens of the
  ControlArena manifest (17,555); Gleipnir at the Section 4 per-token cost
  (one B200, one request at a time), base models at the OOD hosted proxy
  rates plus one output token.
- "About 17,600 tokens" rounds 17,555 (Qwen) and the API tokenizers' 15,455
  to 19,244 lie around it.
- "About twice as long as those of the OOD suite" (2026-11-21): the OOD
  prompts average 9,467 Qwen tokens for Gleipnir 27B (60,539,619 / 6,395)
  and about 8,650 for Gleipnir 4B (55.3M / 6,395), so 17,555 is 1.9 to 2.0
  times as long. Costs follow: Gleipnir 27B on demand $0.767 against $0.414
  (1.9x), Gleipnir 4B $0.190 against $0.094 (2.0x), pplx-decider v1.2
  $0.351 against $0.176 (2.0x).
- Frontier (3% tolerance): on demand Gleipnir 4B → pplx-decider v1.3 → GPT-6.1
  Terra; committed use 4B* → 27B* → v1.3 → Terra (27B* costs 6% less than
  v1.3). Same frontiers on catch rate.
- Paper rows of `tab:sleightbench`: per-trial catch rates read from their
  appendix per-model figures 21–26 (CA prompt, stripped), as recorded in the
  SleightBench finding; Opus 4.6 in the Claude Code scaffold 62%.
- Kimi K3 threshold 0.9841 (highest benign score), catch 0/40.
- Figures `fig:sleightbench` and `fig:sleightbench-catch` plot committed-use
  Gleipnir only and omit the base models and pplx-decider v1.2, superseded
  by v1.3 (JF, 2026-11-19); the caption does not mention them. One frontier:
  Gleipnir 4B → 27B → pplx-decider v1.3 → GPT-6.1 Terra on both metrics. The
  SleightBench appendix sits in Appendix G so that Appendix C holds only
  results that shaped the monitors.

## Jev 1.14 (JF, 2026-11-19)

- `tab:ood-all` row: `results/decision_models_ood/jev_1_14/full/summary.json`
  (0.8763 pAUROC@20, 0.9393 AUROC, $0.3887/1K at $0.042/M) and recall 0.783
  from `results/ood_recall_at_fpr/recall_at_fpr.json`. Registry row added to
  `docs/research/tool_trajectory_ood_frontier.md`.
- `tab:sleightbench` row: run `sb16` (0.260 [0.165, 0.391], AUROC 0.679, catch
  0.200 [0.075, 0.325], $0.771/1K).
- Figures `fig:frontier`, `fig:recall-frontier`, `fig:sleightbench` and
  `fig:sleightbench-catch` show Jev 1.14 in place of Jev 1.13 (superseded);
  the frontiers are unchanged.
- GPT-5.5 is omitted from the SleightBench figures (JF, 2026-11-19): an older
  model scored as a check against the authors' catch rates; the table keeps it.
- "No training, prompt or selection decision of Gleipnir used these benchmarks" moved from
  Section 6.1 to the Section 6 opening (JF); ARK to confirm it holds for the
  benchmarks of their subsections.
- Section 6.1 gives no interval values in the text (JF, 2026-11-19); it
  states that all intervals except GPT-6.1 Terra overlap and points to
  `tab:sleightbench`.

## OpenAI log probabilities (Appendix C.1, JF, 2026-11-19)

- 2026-11-19 probes (unbilled HTTP 400s) on `gpt-6.2-sol` and `gpt-6.2-astra`:
  `reasoning.effort: none` → "'none' is not supported ... Supported values are:
  'low', 'medium', 'high', 'xhigh', and 'max'"; `low` with
  `message.output_text.logprobs` → "logprobs are not supported with reasoning
  models." OpenAI's reasoning guide states that GPT-6 Astra and GPT-6.1 Sol do
  not support `none` (https://developers.openai.com/api/docs/guides/reasoning,
  read 2026-11-19). The Luna canary
  (`docs/findings/openai_luna_logprobs_canary.md`) shows logprobs only with
  effort `none`.
- "Only OpenAI": Anthropic's API returns no log probabilities. Google's
  `responseLogprobs` worked on Gemini 2.x, and developer reports (Google AI
  forum, LiteLLM test matrix) say Gemini 3.x models reject it; we found no
  official Google statement, hence "current models".
- `tab:sleightbench` adds the authors' Claude Code scaffold rows (JF,
  2026-11-19): Haiku 4.5 7%, Sonnet 4.6 46%, Opus 4.7 60%, Opus 4.6 62%, from
  the paper's per-model figures as recorded in the SleightBench finding. The
  scaffold runs the monitor as an agent that can execute code (the paper's
  counting-attack analysis). Section 6.1 now points to Appendix G.1 for the
  comparison instead of quoting Opus 4.6's 32%.
- `tab:ood-all` row spacing 0.96 → 0.94 so the Jev 1.14 row fits the page.
- Section 6.1 interpretation (JF, 2026-11-19). "Matches or exceeds its
  teacher on the OOD suite": 0.951 against 0.908 (lead from one SHADE task,
  `app:shade`); SleightBench gap 0.402 − 0.299 = 0.103. Training-data support:
  `docs/findings/sleightbench_released_monitors.md` (source review of the five
  training sources, none with a monitor in the loop; keyword scan of the
  8,688 teacher-training prompts finds SleightBench mechanisms in roughly
  1–3% of rows or fewer). Decider breadth: v1.2 card ("a large amount of other
  decision data") and the blog's RL-rollout monitoring; the v1.3 card credits
  tasksource. Terra 55% vs authors' Claude Code 60–62% and Gemini 3.1 Pro
  high 54% from `tab:sleightbench`. The earlier "honest mistakes" reading was
  dropped: on our four-category mistake-shaped split (21 attacks) Gleipnir 27B
  AUROC 0.615 matches Kimi K3 0.629, so it does not explain the gap.
- Interpretation trimmed (JF): "matches its teacher" on the OOD suite; the
  keyword-scan and test sentences and the one-pass/10-sample caveat were
  removed. "Our training data contains no attacks built against a monitor"
  rests on the source review in the SleightBench finding (no source put a
  monitor in the loop).
- Terra cost sentence: $15.49/1K from `results/sleightbench_eval/frontier.json`;
  mean output about 7 tokens at $5/M is about $0.04 of it. No cost is
  published for the authors' Claude Code scaffold, so the comparison stays
  qualitative.

## Appendix E: cost of serving pplx-decider (JF, 2026-11-19)

- List prices: `experiments/decision_models_ood/config.json` (Mercury Decide
  and d1 $0.02/M; Microsoft-Decision-1, d1-ultra, Jev $0.042/M) and the Decider
  v1.2 OpenRouter endpoint ($0.02/M, checked 2026-10-22).
- Throughput: `docs/research/tool_trajectory_ood_frontier.md` ("Self-hosting
  it costs far more than its list price") and
  `docs/findings/pplx_decider_ood_benchmark.md`: published fast path, 56,359,499
  tokens in 4,170 s = 13,515 tokens/s on a Runpod B200, 2026-10-19. $7.99/h →
  $0.164/M; $3.44/h → $0.0707/M; break-even at $0.02/M needs 111K / 47.8K
  tokens/s. Flash 30,655 tokens/s → $0.072/M against $0.01/M.
- Gleipnir 27B 50,788 tokens/s at one request at a time (Section 4).
- Bidirectional attention: the `pplx_decider` 0.3.0 runtime runs non-causal
  full attention (pplx-decider OOD finding; the v1.1 card also describes
  non-causal attention). v1.2 and v1.3 both publish `"attention_mode":
  "noncausal_full_attention"` in `decision_config.json` on Hugging Face
  (checked 2026-11-19), so the text names both; Section 6.1 cites the
  architecture difference for v1.3.

## Prefix caching for per-action monitoring (Section 4.1, Appendix E.1 and E.3; JF, 2026-11-20)

- `experiments/prefix_caching_replay/README.md`, run `replay02`
  (`results/prefix_caching_replay/replay02/summary.json`): US-CA-2 B200 with
  AMD EPYC 9555, image `1.2-rc2-zstd` (= `1.2`), one request at a time, 96
  benchmark trajectories, 1,882 checks. Isolated 5.8 s / 23.4 s; every
  action uncached 74.1 s / 285.7 s (12.8× / 12.2×); cached 39.9 s / 77.3 s
  (6.9× / 3.3×); cache hit fraction 88.2% / 86.6%; tokens 14,440,542 against
  1,193,325 (12.1×). Mean cached check 39.9/1882 = 21 ms, 77.3/1882 = 41 ms.
  Isolated throughput 206,289 / 50,978 tokens/s against 204,673 / 50,788 in
  Section 4 (+0.8%, +0.4%). Probe: 4B 49.6 ms plain, 49.2 ms with /metrics
  scrapes. Score drift |cached − uncached| mean 0.0146 / 0.0102, max 0.553 /
  0.175. Costs per 1,000 trajectories from `summary.json`.
- Decider per-action cost: 14,440,542 / 96 × 1,000 × $0.02/M = $3.01;
  isolated 1,193,325 / 96 × 1,000 × $0.02/M = $0.249. Approximate: Decider's
  own prompts (raw state plus question) have somewhat different lengths.
- The earlier US-NC-2 run (Xeon 8568Y+, 89.5K tokens/s for the 4B) is not
  used; see the experiment README.
- Section 4.1 states only the speed-up of caching for every-action checking
  (JF): 285.7 / 77.3 = 3.70 (27B), 74.1 / 39.9 = 1.86 (4B); the multiples of
  one check per trajectory stay in Appendix E.1.
- Section 4.4 "Prefix caching" (JF, 2026-11-20): moved from 4.1 because the
  measurement used the released stack, not stock vLLM. "Lowers the cost of
  each billed input token by the same factors": billed tokens are the
  uncached count (14,440,542) and compute time is the cached time, so the
  cost per billed token falls by uncached/cached time (3.70×, 1.86×). The
  cached-input discount practice: OpenAI lists GPT-6.1 Terra cached input at
  $0.05/M against $1/M (`experiments/sleightbench_eval/config.json`). Only
  pplx-decider's attention is documented (non-causal: v1.1 card, v1.2/v1.3
  `decision_config.json`); Jev's announcement names a new architecture
  without details; the others publish nothing.
- Revised (JF, 2026-11-20): no claim about how providers set base prices.
  Sourced fact only: OpenAI pricing page (https://developers.openai.com/api/docs/pricing,
  checked 2026-10-31, recorded in `experiments/sleightbench_eval/config.json`):
  GPT-6.1 Terra $1.00/M input, $0.05/M cached input (one twentieth). Pricing
  every monitor at one uncached check per trajectory is our choice.

## Drex (Background and Section 4.4; JF, 2026-11-20)

- `dhankhar2026drex`: arXiv:2611.07854, "Towards Foundation Decision Models:
  Drex", v1 2026-11-05, v2 2026-11-09, Nace.AI (models and code under
  huggingface.co/nace-ai, github.com/nace-ai). Only statements read verbatim
  are used: open weights for all four models; SFT with permutation (and
  state-shuffle) consistency terms, then RLAF, a KL-regularized RL objective
  whose reward combines the verified answer with a judge's option
  probabilities; "Drex Max ... is trained from Qwen3.8-27B"; "The model
  removes the causal attention mask in its full-attention layers"; "Drex
  Max's attention pattern and readout adopt the open pplx-decider design";
  "Drex Max cannot share the encoded context between questions". Unverified
  (summary only, not used): data scale, teacher names, prices, latency.
  Drex was not benchmarked here.
- Background revision (JF, 2026-11-20): the Perplexity RL-rollout sentence
  was removed (the blog is still cited in Section 6.1). Drex sentence, from
  a verbatim read of the HTML v2 through Appendix C.2: Section 1 "This
  results in higher accuracy in exchange for increased cost per question"
  (for the non-causal dense backbone with readout, bundled changes); no
  causal-vs-non-causal ablation (Sections 1, 10, 11 list it as future work;
  Section 8.8 ablates RLAF, consistency and vision only; Section 5.1's
  non-causal-ratio probe uses hypernetwork weight edits and reports no
  numbers). Benchmarks: Decision Index 0.4.1 (37 public benchmarks in five
  domains), Decision Bench v4 (1,071 single-question decisions) and the
  Artificial Analysis Decision Index (v2.1 on its site: AA-Paperclip,
  EvalSafe-O*NET-AA, When2Call, AA-Omniscience-Decision, HLE-MultiChoice,
  AA-Symmetry; every task one call on a fixed state; read 2026-11-20 at
  https://artificialanalysis.ai/methodology/decision-index). Section 5.3
  "Because the full-attention layers are non-causal, the representation of
  the state depends on the question, so it cannot be cached across
  questions".
- Geuter et al. kept as support for Gleipnir's causal last-token readout (JF,
  2026-11-20): their readout of answer-letter logits at the last prompt token
  with LoRA fine-tuning on general decision tasks.

## Jev as a trusted monitor (Background; JF, 2026-11-20)

- `t2026jevmonitor`: LessWrong post by Venkat T, 2026-09-18, read in full
  2026-11-20. Jev 1.13 (one Noul question, P(yes) as suspicion) on
  ControlArena's APPS backdoor setting (single-file Python with the problem
  statement): AUROC 0.976 / 0.970 against two honest sets, TPR@1% FPR
  0.878 / 0.853; about $0.044 per 1,000 submissions, 0.34 s median latency;
  no LLM baseline was run (positioned against published o3-mini AUROC
  0.92–0.96); also monitor-directed comments and best-of-N attackers. No
  agent trajectories, no architecture or caching discussion. Hence "no
  published evaluation ... of agent trajectories" in the background.
- `subramanian2026jev`: arXiv:2611.00505 (v1, 2026-10-31), Shreyas Subramanian,
  Scale AI. Verbatim: 14 decision models against four generative judges on
  eleven judge benchmarks; cites AgentRewardBench ("Evaluating automatic
  evaluations of web agent trajectories"). Its AgentRewardBench AUROCs (Table
  8, e.g. pplx-decider 0.921) were seen only in a summary and are not used.
  Searches on 2026-11-20 (decision-model, classifier-monitor and product-name
  terms) found no control paper that evaluates decision models as monitors for
  harmful behaviour; prompt-injection papers on Jev (arXiv 2610.17820,
  2610.18612, 2610.14404) were seen only as snippets.
- Section 4.4 OpenRouter sentence (JF, 2026-11-20): public endpoint listings
  (`https://openrouter.ai/api/v1/models/<id>/endpoints`) read 2026-11-20.
  No caching listed: Jev 1.13 and 1.14 (TypeSafe), pplx-decider v1.2 and
  v1.3:free (Perplexity), Gemini Decision 1 (Google AI Studio),
  Microsoft-Decision-1 (Azure). Caching listed but `input_cache_read` equal to
  the input price: d1, d1-ultra (Liquid), Drex Max (DeepInfra). The
  `supports_implicit_caching` flag is not used as evidence: Drex Section 5.3
  says its state cannot be cached across questions, so the flag is likely a
  provider default. Mercury Decide had no endpoint listed on 2026-11-20, hence
  "the endpoints that we checked".
- OpenAI cached-input ratios (JF, 2026-11-20), from
  `experiments/sleightbench_eval/config.json` (OpenAI pricing page, checked
  2026-09-30 to 2026-10-31): GPT-6 Luna $0.10/$0.01, GPT-6 Sol $2.00/$0.20,
  GPT-5.5 $5.00/$0.50 (one tenth), GPT-6.1 Terra $1.00/$0.05 (one twentieth).
- E.3 correction (JF, 2026-11-20): Decider is not the only decision model with
  public weights and serving code. Hugging Face `nace-ai/drex-max-v1` (public,
  updated 2026-11-17) has the weights, `inference.py`, `code/drex_max/` and
  `decision_config.json`; `nace-ai/drex-v1.5`, `drex-v2.0` and `drex-dlm` are
  public too.
- E.3 Drex serving sentence (JF, 2026-11-20): Hugging Face `nace-ai/drex-max-v1`
  README, `inference.py` and `requirements.txt` (read 2026-11-20): Python path
  on Transformers 5.17 with flash-linear-attention and the `kernels` package,
  tested on an H100 (BF16) and L40S (8-bit), default max length 8,192;
  llama.cpp Q8_0 GGUF via a Nace branch of `llama-server`; Kapitsa, an Ollama
  fork and an affine Q4 checkpoint for Apple silicon. No vLLM/SGLang path and
  no throughput figures.

## Drex Max results (Appendix tables and SleightBench figures; JF, 2026-11-20)

- `tab:ood-all`: `results/decision_models_ood/drex_max/full/summary.json`
  (0.8299 pAUROC@20, 0.9334 AUROC; 37,974,756 billed tokens × $0.12/M =
  $0.7126/1K) and recall 0.566 from `results/ood_recall_at_fpr/`. Not in the
  OOD figures (off the frontier; `MAIN_FIGURE_OMISSIONS`).
- `tab:sleightbench` and `fig:sleightbench(-catch)`: run `sb17`, 0.322
  [0.206, 0.483], AUROC 0.706, catch 0.250 [0.125, 0.400], $1.020/1K at
  $0.12/M on 8,499 billed tokens per row.
- Price: OpenRouter lists $0.06/M with `discount: 0.5`; the undiscounted
  $0.12/M is used (user, 2026-11-20). Billing rule: see the decision-model OOD
  finding (needle probes and tokenizer check, 2026-11-20).
- Kernels table, GDN row (JF, 2026-11-20): the released stack runs FlashInfer's
  `flashinfer.gdn_prefill.chunk_gated_delta_rule` in BF16 with FP32 state and
  gates (`docs/decisions/b200_low_precision_attention_serving.md`,
  `docs/decisions/b200_frost_inference.md`: "FROST FP4 MLP/BF16 FlashInfer GDN");
  `experiments/release_serving/serve.py` sets `GLEIPNIR_FLASHINFER_GDN_CP="auto"`
  (FlashInfer's automatic context-parallel route; `experiments/b200_gdn_route`).
  The direct call on empty-state steps is the "shorter host path" of Section 4.3.
- Section 4.4 synchronous-monitoring sentence (JF, 2026-11-20): per-action costs
  from `results/prefix_caching_replay/replay02/summary.json` (27B cached $1.788
  on demand, $0.770 committed) and the Decider estimate of Appendix E.3 ($3.01
  at $0.02/M on the same prompt tokens). Scoped to the decision models because
  OpenAI's models discount cached input too.

## Appendix F: probing at half depth (JF, 2026-11-20)

- 4B: `experiments/probe_generalization/README.md`, run `full01` (5 seeds,
  frozen validation selection, OOD read once): prompted hard attention e12
  0.7057, soft rolling_w10 e8 0.7126; NTP hard attention e8 0.7092, soft
  rolling_w256 e8 0.7643. Soft wins every family on NTP; 7/9 prompted.
- 27B: `experiments/probe_scale_27b/README.md`, runs `scale27b_full01` (NTP:
  hard 0.8142, soft 0.8481, soft wins every family) and `scale27b_prompted01`
  (hard 0.8406, soft 0.8610; last token hard layer 47 0.8867, soft layer 50
  0.8957). Depth curve of prompted soft last-token probes, OOD 0.30 / 0.56 /
  0.79 / 0.85 / 0.86 / 0.87 at layers 24–44, ID ~0.89–0.92 from layer 32.
  Zero-shot Qwen3.8-27B 0.8903 (parity 0.8901).
- Comparisons: Gleipnir 4B BF16 0.804 and 27B BF16 0.935 from `tab:precision`;
  the probe study's own 27B teacher-prompt reference is 0.9352 (same value).
  Gaps: 0.804 − 0.764 = 0.040; 0.935 − 0.861 = 0.074. LoRA soft vs hard from
  `tab:hard-soft`.
- Cost: `experiments/probe_optimized_serving/README.md` (`thr01`, `thr27b01`):
  truncated + deployed probe 48.5 vs 26.4 rows/s (4B, 1.84×), 12.30 vs 6.31
  NTP / 11.07 vs 5.68 prompted (27B, 1.95×), 128-row batches at concurrency
  128; probe ~1% over truncation. Optimized-stack quality loss: 27B −0.001 to
  −0.014, 4B −0.034 to −0.054 (selected −0.051, −0.054; soft attention family
  −0.034), hence "0.034 to 0.054".
  Measured 2026-10-27/29 on the pre-Gigatoken stack.
- Appendix F citations (JF, 2026-11-20): `xin2020deebert` (ACL 2020, pp.
  2246–2251, arXiv 2004.12993; ACL Anthology 2020.acl-main.204) and
  `liu2020fastbert` (ACL 2020 main.537; authors as listed on the ACL virtual
  site). FastBERT is cited only for what it does (intermediate classifiers
  trained on the same model's final classifier, i.e. self-distillation); its
  comparison with hard labels was not checked, so the text makes no claim
  about it. "Probes ... usually trained on dataset labels" refers to Kramár et
  al. Our contribution as framed: soft targets from an external teacher into
  probes on the unadapted model, compared with LoRA fine-tuning.
- `fig:probe-objective` (JF, 2026-11-20): `scripts/plot_probe_objective_dumbbell.py
  --report` on `results/probe_generalization/full01/ood_{prompted,ntp}.json`
  (the slide/presentation figure `figures/probe_objective_ood_dumbbell.svg`
  without title and footnote, report terms). LoRA seeds from
  `experiments/hard_label_seeds/README.md` (medians: +0.074; the text's 0.076
  is the mean difference of `tab:hard-soft`). 27B per-architecture differences
  from `experiments/probe_scale_27b/README.md` family tables (prompted +0.011
  to +0.058, NTP +0.034 to +0.049, excluding mean-pooled probes).
- `fig:probe-objective-27b` (JF, 2026-11-20): `scripts/plot_probe_objective_dumbbell.py
  --report --model 27b` on `results/probe_generalization/scale27b_prompted01/ood_prompted.json`
  and `scale27b_full01/ood_ntp.json`; LoRA row 0.9044 / 0.9087 (one run each,
  compact instruction, `experiments/qwen38_27b_hard_labels/README.md`);
  references Gleipnir 27B BF16 0.935 (`tab:precision`) and zero-shot 0.8901.
  Seed-median gains: prompted +0.011 to +0.057, bare trajectory +0.034 to
  +0.049, 5/5 seeds in every architecture (the README family table, rounded
  separately, gives +0.058 for prompted attention).
- `fig:lasttoken-depth` (JF, 2026-11-20): `scripts/plot_lasttoken_depth_report.py`
  combining the talk figures `lasttoken_depth_id` (slide-figures a7c1497) and
  `lasttoken_ood_depth` (4ceae29) without their annotations ("most score per
  layer" ring dropped, user). Soft targets throughout (the talk's validation
  curve used hard labels). Validation curve: best C per layer in
  `results/probe_generalization/{full01,scale27b_prompted01}/lasttoken_prompted.json`;
  27B OOD: `lasttoken_depth_prompted.json` (layers 24–44) and the selected layer
  50 (0.8957) in `ood_prompted.json`; 4B OOD at its selected layer 17 (0.7272).
  Hard-label 27B selection layer 47 (0.8867). Gap to Gleipnir 27B BF16:
  0.935 − 0.896 = 0.039. The last-token row was removed from `tab:probes`.

## Section 6.2 reference monitors and Appendix F.3 (JF, 2026-11-22)

- Source: `experiments/evilgenie_api_baselines/` (README with frozen
  protocol and results), `results/evilgenie_api_baselines/summary.json`.
  Kimi K3 on OpenRouter (Wafer endpoint, same settings as its Fireworks OOD
  run; threshold from the Fireworks OOD scores by JF's decision), GPT-6 Luna
  with the settings of `experiments/openai_monitoring_ood/config.json`,
  pplx-decider v1.3 with the frozen Jev Noul question. Thresholds: largest
  score flagging at most 1% of the 3,189 benign strict-OOD rows (Kimi
  0.7773, pplx 0.7525, Luna 1 − 3.6e-8).
- Shared rows (Table 2): reconstructed from ARK's scorer order (positives,
  then negatives and controls shuffled with seed 0), the 27B's 1,312 reached
  rows and the 29,696-token limit under both Gleipnir prompts (standalone
  Qwen tokenizer + 12 chat tokens); reproduces 105 / 999 / 115 and 19 / 74
  too long exactly.
- Table 2 rows (shared rows): pplx-decider v1.3 AUROC 0.904 [0.847, 1.000],
  pAUROC@20 0.614 [0.440, 0.998], flagged 16/36, 1/69, 3/999; GPT-6 Luna
  0.894 [0.825, 0.995], 0.629 [0.438, 0.973], 5/36, 0/69, 0/999; Kimi K3
  0.928 [0.883, 1.000], 0.654 [0.482, 1.000], 34/36, 18/69, 56/999.
- All rows (124 / 1,823): AUROC 0.917, 0.888, 0.889 and pAUROC@20 0.652,
  0.610, 0.597 (Kimi, Luna, pplx). Controls 0/180 each. `arc191_c` AUROC on
  shared rows 0.852, 0.760, 0.850. Luna benign OOD: 8.4% of scores ≥ 0.999.

## Section 6.2 on the released stack, Table 2 and Appendix F.3 (JF, 2026-11-22)

Supersedes the Section 6.2 numbers of the two entries above (ARK's BF16
shared rows and the 2026-11-22 reference-monitor entry) for the report text.

- Gleipnir: `experiments/evilgenie_released_stack/` (README: protocol, run,
  results; Pod `i7xih3yuodf207`), `results/evilgenie_released_stack/summary.json`.
  All 124 hacks / 1,823 legitimate / 180 controls, image 1.2, no length limit
  (longest 111,249 tokens; 144 / 154 rows over 29,696). Thresholds from the
  released-stack OOD runs (all 3,189 benign rows): 4B 0.8355, 27B 0.7982.
- Table 2, all rows: 4B AUROC 0.751 [0.655, 0.925], pAUROC@20 0.283 [0.136,
  0.651], flagged 3/55, 0/69, 4/1,823; 27B 0.905 [0.758, 0.981], 0.673
  [0.498, 0.934], 33/55, 2/69, 20/1,823. Reference monitors on all rows
  (`experiments/evilgenie_api_baselines/`, `summary.json`, key `all`): pplx
  0.889 [0.786, 0.979], 0.597 [0.408, 0.918], 20/55, 1/69, 12/1,823; Luna
  0.888 [0.823, 0.989], 0.610 [0.429, 0.946], 9/55, 0/69, 3/1,823; Kimi 0.917
  [0.840, 0.993], 0.652 [0.471, 0.968], 48/55, 18/69, 110/1,823.
- `arc191_c` AUROC: 4B 0.558, 27B 0.858, Kimi 0.856, Luna 0.756, pplx 0.847.
  Without `arc191_c` (55 / 1,687): 4B 0.874 [0.733, 0.950] / 0.514 [0.312,
  0.756]; 27B 0.919 [0.643, 0.994] / 0.830 [0.333, 0.968]. Bootstrap skipped
  5 of 10,000 draws. Per-problem table (Table F) from the same scores; the
  four problems without hacks have 920 legitimate runs (2 / 4 flagged).
- BF16 comparison on the shared rows (105 / 999): released 4B 0.296, 27B
  0.699 pAUROC@20, against ARK's 0.350 and 0.695.
- Timing: 4B 24,223,003 tokens in 110 s; 27B 25,950,127 in 499 s.
- Kimi K3's own 4 hacks: flagged by no monitor (Gleipnir checked on the
  released-stack scores).
