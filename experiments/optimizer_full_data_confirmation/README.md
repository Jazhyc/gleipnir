# Full-data confirmation of RMS-matched Muon

This experiment confirms the two Muon rates advanced by the rank-64, 25%-data
internal-development screen using all 13,149 training rows and seeds 0, 1, and
2. It trains Muon at `5e-5` and `1e-4`, for six new jobs total, and reuses the
three recipe-identical full-data AdamW `5e-5` rank-64 cells from the completed
adapter-capacity campaign.

## Hypothesis and intervention

The primary hypothesis is that scheduled, RMS-matched Muon improves mean macro
AUROC over AdamW when both use rank 64, two epochs, and the full training set.
The intervention is Muon's base learning rate (`5e-5` or `1e-4`). Seeds, model,
data, objective, rank/alpha, quantization, batch, FLA kernels, scheduler, warmup,
and selected-position logits remain fixed.

The AdamW references were trained at commit
`f9dcc365be6b67ab0bac7abe7d70161f9e36957a` with AdamW `5e-5`, linear scheduling,
3% warmup, two epochs, rank 64 / alpha 128, QLoRA NF4 with double quantization
and BF16 compute, effective batch 32, selected-position logits, and FLA 0.5.2.
Their recorded seeds, row count, recipe, and metrics must match before launch.

## Confirmation and selection rule

Training uses 100% of the training artifact, so the internal development set
from the screen is no longer held out. Evaluation therefore uses the frozen
822-row competition validation artifact as a confirmatory surface. The two Muon
rates were fixed before this evaluation; do not add optimizer rates or retune
from these results. The final competition test remains untouched.

A Muon candidate is eligible only if its paired mean macro-AUROC difference
against AdamW is positive, at least two of three seeds improve, its mean macro
balanced-accuracy regression is no worse than 0.002, and its mean macro-Brier
regression is no worse than 0.002. Select the eligible candidate with highest
mean macro AUROC; retain AdamW if neither passes. Report every replicate,
calibration metric, tie diagnostic, and negative result.

## Stop condition and execution

Stop if train and validation identities overlap, the AdamW references fail the
recipe contract, any seed/rate cell is absent, FLA is unavailable, Muon does not
use `match_rms_adamw`, or an evaluation score is missing. The previous rank-64
Muon screen is the matched largest-rank memory/kernel preflight; the launcher
also fails closed on FLA availability before model import.

```bash
python experiments/optimizer_full_data_confirmation/prepare.py
python experiments/optimizer_full_data_confirmation/run_lambda.py
```

The two Lambda lanes each receive three jobs with seeds and learning rates
crossed across GPUs. Ignored artifacts live under
`results/optimizer_full_data_confirmation/`, with final `replicates.csv`,
`aggregate.csv`, `paired_contrasts.csv`, and `summary.json` under `summary/`.

## Results

All six Muon cells completed on 2026-08-13. Neither candidate passed the frozen
promotion gate, so the selected recipe remains rank-64 AdamW at `5e-5`.

- AdamW `5e-5` reached 0.96254 mean macro AUROC, 0.91389 balanced accuracy,
  and 0.06997 Brier.
- Muon `5e-5` reached 0.96151 mean macro AUROC. Its paired AUROC difference was
  -0.00103 and only one of three seeds improved. Its balanced-accuracy and Brier
  differences were -0.00119 and +0.00018, respectively.
- Muon `1e-4` reached the highest raw mean macro AUROC, 0.96351. Its paired
  difference was +0.00097 and two seeds improved, but balanced accuracy regressed
  by 0.00238 and Brier worsened by 0.00313, exceeding both frozen tolerances.

With only three paired seeds, the AUROC effects remain imprecise: approximate
95% paired t intervals are [-0.00824, +0.00618] for Muon `5e-5` and
[-0.00597, +0.00791] for Muon `1e-4`. The internal-screen advantage therefore
did not replicate robustly at full data. Do not tune more optimizer rates on
this validation surface; retain AdamW for the current recipe.
