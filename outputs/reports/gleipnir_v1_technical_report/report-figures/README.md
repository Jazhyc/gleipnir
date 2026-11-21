# Held-out evaluation figures (Section 6 and Appendix F)

Made with the `alignment-research-figures` skill (`anthro_style`, report
register, scale 0.8). No titles inside the figures: the LaTeX captions carry
the takeaway, as for the other report figures.

- `refresh_data.py` rebuilds `data.json` from the git-ignored stress-test
  results (set `GLEIPNIR_DATA_ROOT` to the checkout that holds `results/`).
- `make_figures.py` reads only `data.json`, writes PNG previews here and vector
  PDFs to `../assets/`, and runs the skill's overlap check.

| Figure | Used in | Source of the values |
| --- | --- | --- |
| `held_out_apps_text` | Figure `fig:apps-text`, Section 6.2 | `results/figures/backdoor_pauroc.json` (task bootstrap, 2,000 draws) |
| `held_out_probe` | Figure `fig:apps-probe`, Appendix F.4 | `results/adversarial_crossed_text_probe/<arm>/report.json` (AUROC, task bootstrap, 10,000 draws) |
| `held_out_lab` | Figure `fig:lab`, Appendix F.8 | `results/adversarial_lab_liars/pauroc20_posthoc.json`; liars flagged from `experiments/adversarial_lab_liars/README.md`; readers from `results/adversarial_is_it_4b/report.json` and `qwen-base/summary.json`; Wilson intervals |
| `held_out_lab_setup` | Figure `fig:lab-setup`, Appendix F.8 | Diagram of `experiments/live-peers/liar-scale` (sprint repository) |

The AISI results figure is not drawn yet: it waits for AISI to confirm that
an arXiv report may show these aggregate results.

```bash
GLEIPNIR_DATA_ROOT=/path/to/gleipnir python refresh_data.py   # only when results change
python make_figures.py                                         # needs matplotlib and the skill
```
