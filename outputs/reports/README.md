# Reports

Keep each written report in `outputs/reports/<report_name>/` as a self-contained
LaTeX document with its own `Makefile`, following the
[presentations conventions](../presentations/README.md): commit the source and
the compiled PDF; keep `build/` ignored. Build with `make` from the report's
directory (needs `latexmk`/`pdflatex`; on Hábrók, put the
`texlive/20230313-GCC-11.3.0` module's `bin/x86_64-linux` on `PATH`; the VPS
has TeX Live 2025 from apt).

- [`sprint_proposal/`](sprint_proposal/): one-page MATS exploration-phase
  proposal for the two-week Gleipnir v1 sprint
  ([proposal.pdf](sprint_proposal/proposal.pdf)). Numbers come from the
  [OOD frontier registry](../../docs/research/tool_trajectory_ood_frontier.md),
  the [27B campaign](../../docs/findings/qwen38_27b_campaign.md) and the
  [vocab-distillation diagnostic](../../experiments/gleipnir27b_vocab_distillation/README.md),
  the [GPT-6 Luna benchmark](../../docs/findings/openai_luna_ood_benchmark.md)
  and the [4B B200 latency baseline](../../docs/decisions/b200_inference_benchmark.md)
  at revision `5ef9e5b`.
- [`gleipnir_v1_technical_report/`](gleipnir_v1_technical_report/): arXiv
  technical report for the v1 monitor release, in the NeurIPS 2026 style's
  preprint mode ([report.pdf](gleipnir_v1_technical_report/report.pdf)).
  Scaffold only; adds `make fast`/`make watch` (precompiled preamble) and
  `make arxiv` (upload bundle).
