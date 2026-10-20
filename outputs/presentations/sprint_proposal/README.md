# Gleipnir Sprint Proposal

Authors: Jeremias Ferrao and Abhinav Kumar.

Short, self-contained Beamer pitch (seven slides) for the two-week sprint. It updates the
[written proposal](../../reports/sprint_proposal/): both monitors are now
release candidates, so the sprint moves straight to stress-testing them for a
public model card. The slides cover a short motivation (monitoring cheap enough to deploy
everywhere) and the core idea (one binary decision and teacher-logit
distillation instead of rationale distillation), the OOD frontier
plot, titled with the cost reduction (Gleipnir 27B beats Gemini 3.1 Pro at 1/188 of its cost),
four candidate sprint avenues, the questions we want feedback on, and logistics (a TA and the schedule around EMNLP).

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
