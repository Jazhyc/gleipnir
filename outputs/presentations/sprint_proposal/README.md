# Gleipnir Sprint Proposal

Author: Jeremias Ferrao.

Short, self-contained Beamer pitch (eight slides) for the two-week sprint. It updates the
[written proposal](../../reports/sprint_proposal/): both monitors are now
release candidates, so the sprint moves straight to stress-testing them for a
public model card. The slides cover the problem, the core idea (one binary decision and
teacher-logit distillation instead of rationale distillation), the OOD frontier
plot, the cost reduction (Gleipnir 27B beats Gemini 3.1 Pro at 1/188 of its cost),
four candidate sprint avenues, and the questions we want feedback on.

Read the compiled deck: [presentation.pdf](presentation.pdf). See
[sources.md](sources.md) for numbers, provenance and pricing conventions.

## Files

- `main.tex`: entrypoint, title, author and date.
- `preamble.tex`: theme, shared with `monitor_injection_results/`.
- `slides.tex`: slide content.
- `assets/ood_frontier.pdf`: vector render of the main OOD frontier figure.
- `sources.md`: provenance and interpretation limits.
- `Makefile`: `make` builds `presentation.pdf`; `make clean` removes `build/`.

All build inputs are inside this directory. On the RUG cluster, run
`module load texlive/20230313-GCC-11.3.0` before `make`.
