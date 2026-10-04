# Monitor injection results

Self-contained Beamer scaffold for communicating the Gleipnir monitor-injection
experiments. The current deck contains a title slide and a draft outline;
results slides and audience-specific framing can be developed here.

The local style uses Beamer's Madrid theme with blue title/header bars, a navy
footer, pale blue content blocks and white slide backgrounds. Madrid's footer
shows the author, presentation title, date and slide number.

## Files

- `main.tex`: document entrypoint, title, author and date.
- `preamble.tex`: local theme, typography and packages.
- `slides.tex`: slide content.
- `assets/`: local figures, tables and associated provenance.
- `Makefile`: build and cleanup commands.
- `build/main.pdf`: generated presentation.

All build inputs are inside this directory. Copy this entire folder to another
machine or upload its TeX files and assets to Overleaf, with `main.tex` as the
main document. No Gleipnir Python environment or repository files are required.

## Build

Install TeX Live or MiKTeX with `latexmk`, `pdflatex`, Beamer, Latin Modern and
`booktabs`, then run:

```bash
make
```

The PDF and auxiliary files are written to `build/`, which is ignored by Git.
To remove those generated files:

```bash
make clean
```

On the RUG cluster, enable the available TeX Live module first:

```bash
module load texlive/20230313-GCC-11.3.0
make
```

Edit the slide content in `slides.tex`, metadata in `main.tex` and visual styling
in `preamble.tex`. Use paths such as `assets/figure.pdf` for local assets. Keep
any bibliography, numerical tables and figure-generation inputs inside this
folder when adding them, and document their provenance here or beside the asset.
