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
| `prompts/output_binary.txt` | `OUTPUT INTERFACE` block of `experiments/tool_trajectory_monitoring/prompts/teacher.txt` |
| `prompts/output_ordinal.txt` | `OUTPUT INTERFACE` block of `.../prompts/teacher_ordinal.txt` |
| `prompts/ordinal_score_bands.txt` | `SCHEMING AND SABOTAGE SPECTRUM` block of `.../prompts/teacher_ordinal.txt`, up to the last score band |
| `prompts/output_binary_reasoning.txt` | `OUTPUT INTERFACE` block of `.../prompts/teacher_binary_reasoning.txt` |
| `prompts/output_ordinal_reasoning.txt` | `OUTPUT INTERFACE` block of `.../prompts/teacher_ordinal_reasoning.txt` |

The five `output_*` and `ordinal_*` files are excerpts, not whole prompts.
| `serving_latency.pdf`, `.svg`, `.json` | `experiments/serving_concurrency/plot.py` on `results/serving_concurrency/summary.json` (stock BF16 length sweeps), with the concurrency sweeps of the agent-trajectory benchmark and the Gigatoken length sweeps: `--replace 4b_fast=results/serving_concurrency/agent03/summary.json:4b_fast_g4096e_giga --replace 4b_bf16=results/serving_concurrency/agent03/summary.json:4b_bf16 --replace 27b_fast=results/serving_concurrency/agent03/summary.json:27b_fast_giga --replace 27b_bf16=results/serving_concurrency/agent03/summary.json:27b_bf16 --replace 4b_fast_native=results/serving_concurrency/giga02/summary.json:4b_fast_g4096e_giga_native --replace 4b_fast_yarn=results/serving_concurrency/giga02/summary.json:4b_fast_g4096e_giga_yarn --replace 27b_fast_native=results/serving_concurrency/giga01/summary.json:27b_fast_giga_native --replace 27b_fast_yarn=results/serving_concurrency/giga01/summary.json:27b_fast_giga_yarn` |
| `ood_frontier.pdf`, `.svg` | `scripts/plot_tool_trajectory_ood_frontier.py --report` on `docs/research/tool_trajectory_ood_frontier.md` |
