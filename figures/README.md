# Figures

This directory contains publication-ready figures generated from canonical
repository data rather than hand-maintained plotting copies.

## Projection changes by teacher target and behavior label

`direction_teacher_by_label_distributions.png` and `.pdf` reproduce the existing
complete original-training census figure used in the
[mechanistic writeup](../docs/writeups/mechanistic_injection_writeup.md).
Two panels show layer-20 trained-minus-base projection changes for harmless
and harmful records, each split at Kimi K3 target 0.5. All 8,688 originals are
included, with identical bins/scales, unit area per curve and dotted group means.
These are descriptive activation changes, not causal training-example influence.

Regenerate the census plots with:

```bash
PYTHONPATH=src:. .venv/bin/python -m experiments.b200_training_clean_direction.plot_distributions
```

The entrypoint reads ignored `results/b200_training_clean_direction/clean_direction01/paired_models.jsonl`
and exports to that directory. Copy this named PNG/PDF pair to `figures/` after
checking its plot receipt. The adjacent `.provenance.json` binds the input,
plotting sources and copied exports; [full findings](../docs/findings/b200_training_direction.md)
document the data and interpretation.

## Matched teacher final-boundary scores

`teacher_final_boundary_histograms.png` and `.svg` compare Qwen3.5-27B-FP8
and Kimi K3 on the same 640 full training trajectories from the frozen
teacher-agreement diagnostic. This source/label-balanced subset is not the
full 8,688-row population. Both panels use identical 0.05-wide bins, axes,
and binary-normalized probabilities; no prefix predictions are included.
The input remains an ignored artifact, with provenance and caveats in
`docs/findings/matched_qwen_kimi_teacher_agreement.md`.

```bash
PYTHONPATH=src .venv/bin/python scripts/plot_teacher_score_histograms.py
```

The script prints the input SHA256 and descriptive summary statistics.

`qwen38_27b_kimi_training_calibration.png` and `.svg` show label reliability
of unadapted Qwen3.8-27B (BF16) and Kimi K3 on all 8,688 training rows, from
`experiments/qwen27b_generations_ood/calibration_train.py`.
`teacher_final_boundary_calibration.png` and `.svg` show reliability diagrams
on these same 640 samples, with 10 equal-width bins, bin counts and descriptive
95% Wilson event-rate intervals. Regenerate with
`PYTHONPATH=src .venv/bin/python -m experiments.teacher_agreement.calibration`.
Detailed pooled/per-source results and 5/10/20-bin ECE sensitivity are written
to `results/teacher_agreement/calibration.json`. The intervals assume independent
samples and do not establish held-out or original-population calibration.

## Strict-filtered 4B and Kimi K3 OOD overlap

`strict_filtered_kimi_k3_ood.png`, `.svg` and `.pdf` compare the strict-filtered
4B candidate with cached Kimi K3 on all 6,395 matched OOD trajectories. Panels
show label-conditioned score distributions, paired score density and paired
differences. The adjacent `.provenance.json` binds input/output hashes and stores
pooled, label and source agreement summaries. Teacher and student use different
prompt interfaces; histogram overlap is distinct from paired decision agreement.
See the [OOD finding](../docs/findings/teacher_disagreement_ood.md).

```bash
PYTHONPATH=src:. .venv/bin/python scripts/plot_strict_kimi_ood.py
```

## Matched 4B/9B training losses

`monitor_training_losses.png`, `.svg` and `.pdf` compare the regular 8,688-row
4B/9B runs: five-update online loss reports, trailing 50-update averages and
whole-population teacher entropy. These are changing-checkpoint training losses,
not a final-checkpoint training evaluation. The adjacent provenance file binds
inputs and exports; [findings](../docs/findings/monitor_training_loss_fit.md)
record the comparison and its capacity/optimization qualifications.

```bash
PYTHONPATH=src:. .venv/bin/python scripts/plot_monitor_training_losses.py
```

## Tool-trajectory OOD frontier

`tool_trajectory_ood_frontier.svg` is the sharp, resolution-independent primary
artifact; a high-resolution PNG is retained for software that does not render
SVG. SVG text is saved as outlines, so it renders identically without the
Cantarell font installed. Both are generated directly from the reference table in
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
uses logits, logprobs, or a native binary probability; other completed
interfaces remain in the registry. Plot labels omit the redundant `base` suffix
without changing canonical model names.
The MoE point uses the presentation label `Qwen3.5-35B-A3B`; provider provenance
is recorded in the canonical registry.
The two mixed-data soft-distilled students use the presentation names
`Gleipnir 4B` and `Gleipnir 9B`; dashed red arrows connect their corresponding
unadapted Qwen3.5 baselines to make the measured adaptation uplift explicit.
The paper's Qwen3.5-4B SFT+RL and Qwen3.5-27B SFT+RL points remain labelled
despite being off the frontier because they anchor matched-backbone comparisons:
Gleipnir 4B shows the distillation uplift at 4B, while Qwen3.5-27B logits shows
the gain from changing the evaluation interface without training the backbone.
A muted blue dotted line recomputes the Sinha et al. frontier from their visible
points; the solid dark line is the combined frontier after adding our results.
Their separation at low cost shows which historical frontier points the
Gleipnir students displace.

GPT-6 Luna is included at `$0.8355/1K` and `0.8365` Mean-OOD pAUROC@20,
using the accepted responses' token usage and ordinary uncached Standard rates.
Retries and cache writes are excluded under the registry contract. It dominates
Gleipnir 9B at these dated prices; the 9B point and
its matched adaptation arrow remain visible. The plot supports actual returned
binary token logprobs and native probabilities as well as logits. GPT-5.6 Luna
is not included.

Jev 1.13 is included from the complete retained native Decisions benchmark at
`$0.3887/1K` and `0.8495` Mean-OOD pAUROC@20. It enters the frontier between
Gleipnir 4B and Kimi K2.6, dominating GPT-6 Luna and Qwen27B as well as
Gleipnir 9B. All comparison points remain visible. The native state/question
interface and two provider-usage/context discrepancies are documented in
`docs/findings/tool_trajectory_jev_ood_benchmark.md`.

pplx-decider v1.2 (27B, labelled `Decider v1.2`) is included at `$0.1763/1K` and `0.8956` Mean-OOD
pAUROC@20. It was scored locally on the Jev requests and priced at its OpenRouter
list rate. It is the lowest-cost frontier point and dominates Gleipnir 4B and
Jev 1.13; both remain visible. Its Flash sibling (`0.5350`) falls below the
plotted range and is not in the registry; see
`docs/findings/pplx_decider_ood_benchmark.md`.

Pass `--source` to override the registry. Pass `--output` one or more times to
select custom output paths and formats, for example:

```bash
python scripts/plot_tool_trajectory_ood_frontier.py \
  --output /tmp/frontier.svg --output /tmp/frontier.png
```

For future figures, reuse `gleipnir.plotting.set_plot_style`,
`pareto_frontier_mask`, and `save_figure` so typography, colors, and export
behavior remain consistent.

## Qwen3.5-9B OOD data scaling

`tool_trajectory_ood_data_scaling.svg` plots the six matched-data Qwen3.5-9B
soft-distillation checkpoints from 204 through 8,688 training examples. It
shows only source-balanced Mean-OOD pAUROC@20 and overlays the Qwen3.5-4B and
Qwen3.5-27B rationale-SFT scaling curves from Sinha et al. Figure 7. The paper
values were recovered from the original vector paths, rounded to three
decimals, and recorded with provenance in the canonical research note. The
figure explicitly warns that supervision, optimizer budgets, checkpoint
selection, subsets, and output interfaces differ. The mixed-data Gleipnir 9B
and 4B transfer conditions are intentionally excluded.

The SVG and PNG are generated directly from the canonical results table in
`docs/research/tool_trajectory_distillation_ood_scaling.md`:

```bash
python scripts/plot_tool_trajectory_ood_scaling.py
```

As with the frontier plot, pass `--source` to override the canonical Markdown
table and repeat `--output` to select one or more custom output formats.

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

## SLEIGHT-Bench catch rate and pAUROC@20

`sleightbench_catch_rate.png`/`.svg` and `sleightbench_pauroc.png`/`.svg`
show stripped-view SLEIGHT-Bench results as grouped bars (one group per
model, one bar per way of running it) for the
[writeup](../docs/writeups/sleightbench_monitor_evaluation.md). Values are
hard-coded from the [finding](../docs/findings/sleightbench_released_monitors.md):
paper rows are per-trial catch rates read from the paper's figures; Gleipnir
rows are the optimized-stack `sb02` runs and base-model logit rows the BF16
`sb04` runs. Regenerate with:

```bash
PYTHONPATH=src .venv/bin/python scripts/plot_sleightbench_results.py
```

## Probe objective: hard labels versus Kimi soft targets (OOD)

`probe_objective_ood_dumbbell.png`/`.svg` compare hard-label and Kimi
soft-target training on the frozen OOD suite (source-macro pAUROC@20, seed
median, individual seeds as small dots). Rows are each probe family's ID-best
candidate on the prompted ("monitor prompt") and NTP ("raw transcript")
backbones (`full01`, 5 seeds), plus the matched 4B LoRA seeds 0–2 as the
fine-tuned reference. Stars mark each objective's frozen ID-rule selection,
which is always one of the family rows. Each Δ badge is shaded by sign, next to
the count of same-seed pairs where soft wins. Mean-pooled probes (near the
floor) and the single-fit last-token probes (no seeds to pair) are left off the
plot; the footnote gives their scores, the
[probe study](../experiments/probe_generalization/README.md) reports them in
full, and the [hard-label seeds](../experiments/hard_label_seeds/README.md)
supply the hard-coded LoRA values. The probe inputs are ignored artifacts.
Regenerate with:

```bash
PYTHONPATH=src .venv/bin/python scripts/plot_probe_objective_dumbbell.py
```

Draft caption (carries the pre-registered result the figure leaves to the
stars):

> Hard labels versus Kimi K3 soft targets on the out-of-distribution suite.
> Probes are small classifiers trained on frozen Qwen3.5-4B activations
> (layer-15 residual stream), computed either with the transcript inside the
> monitor prompt or on the raw transcript alone; the fine-tuned monitor is the
> Gleipnir 4B LoRA recipe. Choosing each target's probe on the validation set
> (stars), soft targets add +0.007 OOD pAUROC@20 with the monitor prompt and
> +0.055 on the raw transcript, against +0.074 for the fine-tuned monitor.
> pAUROC@20 is the normalized area under the ROC curve up to a 20%
> false-positive rate.
