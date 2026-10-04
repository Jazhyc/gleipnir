# Emergent Suspiciousness in Trained Monitors

Authors: Jeremias Ferrao and Michael Kenney.

Subtitle: Prompt Injection Shortcuts, Context-Dependent Robustness, and Data Augmentation.

Self-contained Beamer presentation for communicating monitor-injection experiments.
The current opening has ten main content slides plus the title for a five-minute talk, followed by
four backup slides.
It contrasts rationale SFT plus RL with our direct binary-logit supervision and
shows both methods' mean OOD pAUROC@20 at 4B: 0.684 versus 0.776. The slide uses
a paper citation accessible to the audience, without formulas or references to
repository files. The next slide motivates testing monitor-directed injections,
illustrates Terekhov et al.'s APPS benchmark with a short injection excerpt
beside one blue results box centered vertically against the code excerpt.
It then leads into testing our trained monitors against these injections,
with the original paper citation. The initial-results slide reports near-perfect injected-backdoor detection at
4B and 9B, then asks whether injection text itself drives the alarms.
The next slide answers with harmless-code controls: flag rates rise from
1.98% on clean code to 100.00% / 99.85% with injections at the same thresholds.

Detailed source counts, training hyperparameters and standalone OOD comparison
slides are omitted from this short introduction. The OOD chart assets remain
available as supporting material. See [sources.md](sources.md) for provenance,
input caveats, metric definitions and comparison limits.

Read the compiled deck: [presentation.pdf](presentation.pdf). This PDF is
committed alongside its source so readers can view it without installing LaTeX.

The local style uses Beamer's Madrid theme with blue title/header bars, a navy
footer, pale blue content blocks and white slide backgrounds. Madrid's footer
shows the author, presentation title, date and slide number.

## Files

- `main.tex`: document entrypoint, title, author and date.
- `preamble.tex`: local theme, typography and packages.
- `slides.tex`: slide content.
- `script.md`: six-minute read-aloud script with slide cues and timing checkpoints.
- `assets/`: local figures, tables and associated provenance.
- `sources.md`: source attribution, artifact identities and interpretation limits.
- `Makefile`: build and cleanup commands.
- `presentation.pdf`: committed, publicly readable presentation.
- `build/`: ignored auxiliary compilation files.

All build inputs are inside this directory. Copy this entire folder to another
machine or upload its TeX files and assets to Overleaf, with `main.tex` as the
main document. No Python environment or repository files are required.

## Build

Install TeX Live or MiKTeX with `latexmk`, `pdflatex`, Beamer, Latin Modern and
`booktabs`, then run:

```bash
make
```

The build produces one PDF, `presentation.pdf`, and keeps auxiliary files in the
ignored `build/` directory. Commit the updated PDF together with its source
whenever the presentation changes. To remove auxiliary files while
keeping the published PDF:

```bash
make clean
```

On the RUG cluster, enable the available TeX Live module first:

```bash
module load texlive/20230313-GCC-11.3.0
make
```

Edit the slide content in `slides.tex`, metadata in `main.tex` and visual styling
in `preamble.tex`. Use paths such as `assets/figure.png` for local assets. Keep
any bibliography, numerical tables and figure-generation inputs inside this
folder when adding them, and document their provenance here or beside the asset.

The checked-in charts need no Python to build the PDF. To regenerate them,
install Matplotlib (3.11.1 was used) and run `python assets/plot_ood_comparison.py`.
The script reads only the local aggregate JSON and imports no project code.

The JudgeDeceiver slide shows original-adapter A/B quality-judging accuracy
for clean answers and injections on better or worse answers, demonstrating
avoidance in a different task without treating preference labels as harm labels.

Visible slides use “adapter” or “original” for the initial models; “regular”
is retained only in provenance records as the internal condition identifier.

The deployment-risk slide explains how injection-text false alarms could
create review overload and denial of service when alarms gate autonomous work.

The first diagnosis slide tests broad training-data filtering: four explicitly
tagged injection objectives, 1,154 model-flagged exclusions, and unchanged
100% APPS injected-harmless alarms after retraining the 4B adapter.

Diagnosis 2 summarizes Michael’s collaborator-reported Kimi K3 teacher scoring:
template-dependent hard verdicts and strong matched AUROC challenge a simple
inheritance explanation. The accompanying finding records audit limitations.

Diagnosis 3 tests conservative instruction framing with unchanged training data
and teacher targets; injected-harmless APPS alarms remain 100%. It cites the
emergent-misalignment framing work that motivated the intervention.

Intervention 1 presents joint 75:25 monitoring/JudgeDeceiver training: strong
held-out quality judgments, but unchanged APPS injected-harmless alarms.

Intervention 2 shows direct monitoring augmentation: 40% replacement with
injected views and reused teacher targets. At 4B, matched APPS injected AUROC
rises to 0.859 while injected-harmless flag rate falls to 44.43%.

The augmentation table compares both 4B and 9B, including injected-backdoor
TPR at clean-code ~2% FPR, emphasizing monitor-dependent benefits and tradeoffs.

The following 4B transfer slide compares clean ID monitoring and JudgeDeceiver
quality judging, showing broader gains with substantial residual avoidance.

The takeaways slide covers unexpected generalization, broader augmentation,
unresolved mechanisms and limited replication.

## Five-minute flow and backup

The main talk keeps the methods, benchmark motivation, initial findings,
JudgeDeceiver diagnostic, one summary of the three causal hypotheses, both
interventions, transfer results and takeaways. The DoS scenario and the detailed
data-filtering, teacher-inheritance and instruction-framing slides follow the
takeaways in an unnumbered backup section. The summary narrows explanations
without claiming a causally identified training mechanism.

Suggested time budget (seconds): title 5; methods 35; benchmark 25; initial
results 20; harmless controls 20; judging 25; diagnostic summary 25; auxiliary
training 30; augmentation 45; transfer 35; takeaways 35. Total: 300 seconds.
Backup slides are for questions and are excluded from the main frame count.
