# Emergent Suspiciousness in Trained Monitors

Authors: Jeremias Ferrao and Michael Kenny.

Subtitle: Prompt Injection Shortcuts, Context-Dependent Robustness, and Data Augmentation.

Self-contained Beamer presentation for communicating monitor-injection experiments.
The current opening explains the inherited trajectory data, contrasts rationale
SFT plus GRPO with binary logit distillation, and describes our regular 4B
adapter recipe. Two charts compare it with Sinha et al.'s published 4B SFT+RL
results on the six-source OOD suite. The final opening slide introduces the
prompt-injection stress tests; their results narrative remains to be added.

The OOD gain is substantial on the source mean, with an AgentDojo regression.
The slides distinguish a recipe comparison from an objective-only ablation and
state the reasoning-bearing training-input caveat. See [sources.md](sources.md)
for exact provenance, metric definitions and comparison limits.

Read the compiled deck: [presentation.pdf](presentation.pdf). This PDF is
committed alongside its source so readers can view it without installing LaTeX.

The local style uses Beamer's Madrid theme with blue title/header bars, a navy
footer, pale blue content blocks and white slide backgrounds. Madrid's footer
shows the author, presentation title, date and slide number.

## Files

- `main.tex`: document entrypoint, title, author and date.
- `preamble.tex`: local theme, typography and packages.
- `slides.tex`: slide content.
- `assets/`: local figures, tables and associated provenance.
- `sources.md`: source attribution, artifact identities and interpretation limits.
- `Makefile`: build and cleanup commands.
- `presentation.pdf`: committed, publicly readable presentation.
- `build/`: ignored auxiliary compilation files.

All build inputs are inside this directory. Copy this entire folder to another
machine or upload its TeX files and assets to Overleaf, with `main.tex` as the
main document. No Python environment or repository files are required.

## Build

Install TeX Live or MiKTeX with `latexmk`, `pdflatex`, Beamer, Latin Modern and
`booktabs`, then run:

```bash
make
```

The build produces one PDF, `presentation.pdf`, and keeps auxiliary files in the
ignored `build/` directory. Commit the updated PDF together with its source
whenever the presentation changes. To remove auxiliary files while
keeping the published PDF:

```bash
make clean
```

On the RUG cluster, enable the available TeX Live module first:

```bash
module load texlive/20230313-GCC-11.3.0
make
```

Edit the slide content in `slides.tex`, metadata in `main.tex` and visual styling
in `preamble.tex`. Use paths such as `assets/figure.png` for local assets. Keep
any bibliography, numerical tables and figure-generation inputs inside this
folder when adding them, and document their provenance here or beside the asset.

The checked-in charts need no Python to build the PDF. To regenerate them,
install Matplotlib (3.11.1 was used) and run `python assets/plot_ood_comparison.py`.
The script reads only the local aggregate JSON and imports no project code.
