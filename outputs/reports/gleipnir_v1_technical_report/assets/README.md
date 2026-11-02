# Report assets

Figures included by the report (`\includegraphics{<name>}` resolves here).
Use vector PDFs for plots, exported from the portable SVG sources. pdfLaTeX
cannot include SVG. Keep each plotting script and its small input data next to
the figure, and record the source artifact and revision in
[../sources.md](../sources.md). `make arxiv` ships only `.pdf`, `.png` and
`.jpg` files from this directory.
