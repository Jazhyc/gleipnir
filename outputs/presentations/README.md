# Presentations

Keep each presentation in `outputs/presentations/<presentation_name>/`, using
LaTeX/Beamer. This is a public-facing output: commit the compiled PDF alongside
its source so readers can understand the project without building the slides.
Every presentation must build after its folder is copied out of
this repository. Include its own slides, preamble, assets, references and build
instructions; avoid symlinks or inputs from elsewhere in the repository.

The first deck is [`monitor_injection_results/`](monitor_injection_results/).
It contains a title and a single methods introduction for a six-minute talk;
the injection-results narrative is still being developed.
The compiled deck is [presentation.pdf](monitor_injection_results/presentation.pdf).

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
    ├── presentation.pdf     # compiled public deck; committed
    ├── assets/
    └── build/               # auxiliary compilation files; ignored
```

Build a presentation from its own directory:

```bash
cd outputs/presentations/monitor_injection_results
make
```

This requires a TeX Live or MiKTeX installation providing `latexmk`, `pdflatex`
and the packages listed in the presentation README. `make` publishes the final
PDF as `presentation.pdf`; it is the only compiled deck. Each folder's
`make clean` removes its auxiliary `build/` directory and preserves the
published PDF.

For a new presentation, copy the initial scaffold to a new folder, remove its
copied `build/` directory with `make clean`, and update the title, slides and
README. Rebuild the new deck to replace the copied PDF. Track source files, small
presentation assets and the final PDF; keep intermediate build files ignored.
Update and commit the PDF with each presentation change. Record the source
revision and metric definitions for reported results, and include local asset
provenance or references where applicable.
