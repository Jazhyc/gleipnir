# Figures

This directory contains publication-ready figures generated from canonical
repository data rather than hand-maintained plotting copies.

## Tool-trajectory OOD frontier

`tool_trajectory_ood_frontier.png` is generated directly from the reference
table in `docs/research/tool_trajectory_ood_frontier.md`. The script validates
the table's declared frontier against recomputed point-estimate nondominance
before plotting:

```bash
python scripts/plot_tool_trajectory_ood_frontier.py
```

Pass `--source` or `--output` to override either path. The output extension
selects the image format, so an SVG can be generated with:

```bash
python scripts/plot_tool_trajectory_ood_frontier.py \
  --output figures/tool_trajectory_ood_frontier.svg
```

For future figures, reuse `gleipnir.plotting.set_plot_style`,
`pareto_frontier_mask`, and `save_figure` so typography, colors, and export
behavior remain consistent.
