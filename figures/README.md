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
The Qwen3.5-9B reasoned-binary ablation likewise remains in the registry but is
omitted because it nearly overlaps the immediate Qwen3.5-9B point and adds
visual noise without changing the frontier.
Models below `0.60` Mean-OOD pAUROC@20 are also retained in the registry but
omitted from this presentation so the operationally relevant upper frontier has
room for future low-cost, high-performing monitors.
Gleipnir points are included only when their evaluation interface explicitly
uses logits; other completed Gleipnir interfaces remain in the registry. Plot
labels omit the redundant `base` suffix without changing canonical model names.

Pass `--source` to override the registry. Pass `--output` one or more times to
select custom output paths and formats, for example:

```bash
python scripts/plot_tool_trajectory_ood_frontier.py \
  --output /tmp/frontier.svg --output /tmp/frontier.png
```

For future figures, reuse `gleipnir.plotting.set_plot_style`,
`pareto_frontier_mask`, and `save_figure` so typography, colors, and export
behavior remain consistent.

## Distillation pipeline comparison

`distillation_pipeline_comparison.svg` contrasts Sinha et al.'s filtered
rationale-distillation pipeline with Gleipnir's direct binary
decision-distribution distillation. The diagram keeps the shared action-only
input and deployable-monitor endpoint aligned while showing the paper's
candidate generation, label-conditioned retry, external judging, and
best-candidate selection stages. It intentionally omits optimizer and epoch
details, which belong in the methods text rather than the pipeline figure.

Regenerate the vector and raster artifacts with:

```bash
python scripts/plot_distillation_pipeline_comparison.py
```
