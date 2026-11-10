# Report assets

Figures included by the report (`\includegraphics{<name>}` resolves here).
Use vector PDFs for plots, exported from the portable SVG sources. pdfLaTeX
cannot include SVG. Keep each plotting script and its small input data next to
the figure, and record the source artifact and revision in
[../sources.md](../sources.md). `make arxiv` ships only `.pdf`, `.png`,
`.jpg` and `.txt` files from this directory.

`prompts/` holds byte copies of the instructions that the appendix prints with
`\VerbatimInput`. Do not edit them here; copy them again from their sources if
those change, and update the hashes in [../sources.md](../sources.md).

| File | Source in the repository |
| --- | --- |
| `prompts/compact_instruction.txt` | `experiments/tool_trajectory_monitoring/prompts/student.txt` |
| `prompts/full_specification.txt` | `experiments/tool_trajectory_monitoring/prompts/teacher.txt` |
| `prompts/control_arena_binary.txt` | `experiments/sleightbench_eval/prompts/control_arena_binary.txt` |
