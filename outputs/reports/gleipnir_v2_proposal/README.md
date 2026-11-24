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

## Sources of numbers

- v1 results: the [v1 report](../gleipnir_v1_technical_report/) and its `sources.md`;
  3,189 benign trajectories is the benign count of the six-benchmark suite there.
- Inkling 2 Mini: [experiments/inkling_ood](../../../experiments/inkling_ood/README.md).
- SWE-chat context lengths: `max_ctx` (largest per-call agent context, input plus
  cache tokens) of `data/swe_chat/per_session.parquet` (revision `202b071f`; see
  its `PROVENANCE.md`), 17,819 sessions: 74.8% over 28,000 tokens, 13.7% over
  200,000, median 77,367, maximum 1,056,358.
- PrefillOnly and TokenGate figures: the papers' abstracts and evaluation
  sections (arXiv 2505.07203v1 §7; 2610.26754v1 abstract and §5). TokenGate's
  1.4–11.3× is time to first token against full-context prefill on Qwen3-1.7B
  tasks; 3.8× is its hybrid model (Qwen3.5-2B).
