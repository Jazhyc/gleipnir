# Presentations

Keep each presentation in `outputs/presentations/<presentation_name>/`, using
LaTeX/Beamer. Every presentation must build after its folder is copied out of
this repository. Include its own slides, preamble, assets, references and build
instructions; avoid symlinks or inputs from elsewhere in the repository.

The initial scaffold is [`monitor_injection_results/`](monitor_injection_results/).
It contains a title slide and a draft outline, ready for the results narrative.

```text
outputs/presentations/
├── README.md
└── monitor_injection_results/
    ├── README.md
    ├── Makefile
    ├── .gitignore
    ├── main.tex
    ├── preamble.tex
    ├── slides.tex
    ├── assets/
    └── build/                # generated PDF and compilation files; ignored
```

Build a presentation from its own directory:

```bash
cd outputs/presentations/monitor_injection_results
make
```

This requires a TeX Live or MiKTeX installation providing `latexmk`, `pdflatex`
and the packages listed in the presentation README. Each folder's `make clean`
removes its generated `build/` directory.

For a new presentation, copy the initial scaffold to a new folder, remove its
copied `build/` directory with `make clean`, and update the title, slides and
README. Track source files and small presentation assets; keep generated build
files ignored. Record the source revision and metric definitions for reported
results, and include local asset provenance or references where applicable.
