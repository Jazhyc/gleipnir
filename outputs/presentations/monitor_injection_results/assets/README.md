# Presentation assets

Supporting OOD numerical inputs and charts live here. They are retained for
reference but are not shown in the two-slide methods opening:

- `ood_comparison.json`: exact local aggregate metrics, published baseline
  values and reported standard deviations, sample counts, and artifact hashes.
- `plot_ood_comparison.py`: standalone Matplotlib chart generator.
- `ood_mean.png`: equal-source mean OOD pAUROC@20 comparison.
- `ood_by_source.png`: all six OOD sources, including AgentDojo's regression.

The pale bars represent Sinha et al.'s published Qwen3.5-4B SFT+RL baseline;
blue bars represent our regular trajectory-only 4B logit-distilled adapter.
Error bars reproduce the paper's reported SD, not a confidence interval.
Our single-checkpoint scores have no uncertainty bars.

See the local `../sources.md` for attribution, metric definition, training
provenance and comparison limits. These PNG charts are original renderings of
aggregate results; no raw dataset, checkpoint or paper figure is included.

Regenerate with Python 3.12 and Matplotlib 3.11.1 (the version used here):

```bash
python assets/plot_ood_comparison.py
```

Run from the presentation directory; the script resolves its own local paths
and also works from any other working directory. The checked-in PNGs allow
LaTeX builds without Python or Matplotlib.
