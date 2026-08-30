# Figures

This directory contains publication-ready figures generated from canonical
repository data rather than hand-maintained plotting copies.

## Tool-trajectory OOD frontier

`tool_trajectory_ood_frontier.svg` is the sharp, resolution-independent primary
artifact; a high-resolution PNG is retained for software that does not render
SVG. Both are generated directly from the reference table in
`docs/research/tool_trajectory_ood_frontier.md`. The script validates the
table's declared frontier against recomputed point-estimate nondominance before
plotting:

```bash
python scripts/plot_tool_trajectory_ood_frontier.py
```

Claude Opus 4.6 remains in the canonical registry but is omitted from the plot:
its `0.002` pAUROC@20 gain over Gemini 3.1 Pro comes at substantially higher
inference cost and crowds the upper endpoint without changing the comparison.

Pass `--source` to override the registry. Pass `--output` one or more times to
select custom output paths and formats, for example:

```bash
python scripts/plot_tool_trajectory_ood_frontier.py \
  --output /tmp/frontier.svg --output /tmp/frontier.png
```

For future figures, reuse `gleipnir.plotting.set_plot_style`,
`pareto_frontier_mask`, and `save_figure` so typography, colors, and export
behavior remain consistent.
