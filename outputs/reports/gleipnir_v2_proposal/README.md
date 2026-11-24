# Gleipnir v2 funding proposal

Funding proposal for version 2 of the Gleipnir monitors (sole investigator,
about six weeks from December 2026 to mid-January 2027; theme: production
readiness on Inkling 2 Mini), to be shared with AI
safety funders. Status: template; every section is a `\todo{}` stub.

It cites the version 1 technical report
([source](../gleipnir_v1_technical_report/)) as `ferrao2026gleipnir`, with the
same BibTeX key as the project page. Its URL is the project page; switch it to
the arXiv identifier once v1 has one.

## Files

- `main.tex`: entrypoint, title block and section order.
- `preamble.tex`: layout, packages and the `\todo{}` macro.
- `sections/`: one file per section.
- `references.bib`: BibTeX entries (`unsrtnat` style, numeric natbib citations).
- `assets/`: figures (create when needed).
- `proposal.pdf`: committed compiled proposal.
- `build/`: ignored build output.

## Build

```bash
make          # -> proposal.pdf
make watch    # rebuild on every save
make check    # page count, \todo{} markers, undefined references, overfull boxes
make clean    # remove build/
```

Before sending, run `make check` and resolve every marker.
