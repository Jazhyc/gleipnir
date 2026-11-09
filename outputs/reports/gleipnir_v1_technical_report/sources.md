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
- Kutasov et al., *SHADE-Arena: Evaluating Sabotage and Monitoring in LLM
  Agents*, [arXiv:2506.15740](https://arxiv.org/abs/2506.15740). Cited for
  the terms main task and side task; only the abstract was read (2026-11-09).

BibTeX metadata for all entries comes from the arXiv API (read 2026-11-09).
