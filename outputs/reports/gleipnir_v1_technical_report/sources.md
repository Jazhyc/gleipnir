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
  architectures in Section 6.
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

BibTeX metadata for all entries comes from the arXiv API (read 2026-11-09).
