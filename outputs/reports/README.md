# Reports

Keep each written report in `outputs/reports/<report_name>/` as a self-contained
LaTeX document with its own `Makefile`, following the
[presentations conventions](../presentations/README.md): commit the source and
the compiled PDF; keep `build/` ignored. Build with `make` from the report's
directory (needs `latexmk`/`pdflatex`; on Hábrók, put the
`texlive/20230313-GCC-11.3.0` module's `bin/x86_64-linux` on `PATH`).

- [`sprint_proposal/`](sprint_proposal/): one-page MATS exploration-phase
  proposal for the two-week Gleipnir v1 sprint
  ([proposal.pdf](sprint_proposal/proposal.pdf)). Numbers come from the
  [OOD frontier registry](../../docs/research/tool_trajectory_ood_frontier.md),
  the [27B campaign](../../docs/findings/qwen38_27b_campaign.md) and the
  [vocab-distillation diagnostic](../../experiments/gleipnir27b_vocab_distillation/README.md)
  at revision `5ef9e5b`.
