# Gleipnir v1 technical report

arXiv technical report for the version 1 release of the Gleipnir monitors.
Self-contained LaTeX source using the official NeurIPS 2026 style in
`preprint` mode: authors shown, no line numbers, a "Technical report." footer and no
NeurIPS checklist. Title: *Lightweight Restraints for Powerful Agents:
Effective AI Control Monitors at a Fraction of the Cost*. It reads literally on
first pass and also alludes to the myth: Gleipnir is the thin fetter that bound
Fenrir after iron chains failed.

Status: scaffold. Every section is a `\todo{}` stub; draft claims are in
[OUTLINE.md](OUTLINE.md). The main text is limited to nine pages, up to the
end of the conclusion; everything else goes to the appendix.

Read the compiled report: [report.pdf](report.pdf).

## Files

- `OUTLINE.md`: claims per section, agreed before drafting, and open decisions.
- `STYLE.md`: writing rules for all report text; follow them when drafting.
- `main.tex`: entrypoint, title, authors, section order and bibliography.
- `preamble.tex`: style options, packages and the `\todo{}` macro.
- `sections/`: one file per section, plus `appendix.tex`.
- `references.bib`: BibTeX entries (`plainnat` style, natbib citations).
- `neurips_2026.sty`: unmodified official style, from
  `https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`
  (revision January 2026; SHA-256 `c3fc2894…a4555a`).
- `assets/`: figures and their provenance; see [assets/README.md](assets/README.md).
- `sources.md`: source revisions, metric definitions and interpretation limits for reported numbers.
- `report.pdf`: committed compiled report.
- `build/`, `arxiv/`, `arxiv.tar.gz`: ignored build outputs.

## Build

Needs TeX Live with `latexmk`, `pdflatex`, `bibtex`, `cm-super` and the
`latex-extra`/`science` collections (TeX Live 2025 is installed on the VPS).

```bash
make          # clean standard build -> report.pdf
make fast     # same output; reuses a precompiled preamble (about 2x faster per edit)
make watch    # fast build on every save
make check    # main-text pages (limit 9), \todo{} markers, undefined references, overfull boxes
make arxiv    # arxiv.tar.gz: TeX, style, main.bbl and figure files
make clean    # remove build/ and the arXiv bundle; keeps report.pdf
```

Build speed:

- `pdflatex` is the fastest engine for this style. Avoid `fontspec`, so the report doesn't need LuaLaTeX/XeLaTeX.
- `latexmk` reruns only the passes and BibTeX runs a change needs.
- `make fast` dumps everything in `main.tex` up to `\endofdump` into
  `build/preamble.fmt` (via `mylatexformat`), so each pass skips loading
  packages. The format is rebuilt when `main.tex`, `preamble.tex` or the style changes.
  `\endofdump` is a no-op in normal builds, Overleaf and arXiv. `make watch`
  builds the format only at startup: restart it after editing the preamble,
  and avoid running `make`/`make clean` alongside it, since they share `build/`.
- Include figures as vector PDFs (export from the SVG sources) or compressed PNGs. Large rasters dominate compile time.

On this 4 vCPU VPS, the scaffold takes 3.4 s to build from clean, about 1.0 s
per edit with `make` and about 0.5 s per edit with `make fast`.

## arXiv upload

Run `make check` and resolve every marker, then `make arxiv` and upload
`arxiv.tar.gz`. The bundle ships `main.bbl`, so arXiv doesn't need to run
BibTeX. It excludes `references.bib`, provenance files and figure sources.
Before submitting, check the PDF that arXiv compiles against `report.pdf`.
