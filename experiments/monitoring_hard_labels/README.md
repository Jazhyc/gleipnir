# Tool-trajectory hard-label strength screen

Hypothesis: adding source hard labels to the fixed Kimi K3 targets improves ID
ranking without unacceptable source or calibration regression. Earlier completed
tool-trajectory students used soft targets only; previous hard-label anchoring
results belong to the deception domain and do not establish a monitoring gain.

Freeze four seed-0 fractions: 0 (fresh soft-only control), 0.1, 0.3 and 1
(hard-only endpoint). The objective is `(1-alpha)*soft_BCE + alpha*hard_CE`
at the same selected literal 0/1 logits; hard CE equals binary BCE with source
labels. The weights sum to one so this does not increase overall loss scale.
Keep source labels and teacher probabilities separate, including for hard-only.
Use all 8,688 monitoring trajectories and no deception rows, one epoch (272
updates), AdamW LR 2e-5, linear decay, 3% warmup, no dropout or weight decay,
rank 128/alpha 256 and exactly the same initial FP32 master in every condition.
Preserve the regular compact student instruction and reasoning-bearing training
inputs; this screen does not alter prompts or clean training reasoning.

Use the selected `qwen35_4b_b200_default` packed BF16 profile: logical batch 32,
16,384-token packing budget, no model checkpointing, selected-token projection,
uniform pinned FlashQLA, segmented causal SDPA and selective shell compilation.
Reuse the completed regular-4B recipe receipt at the user’s explicit request;
skip repeated kernel/packing/partition diagnostics and memory preflight. Record
reference identity and skipped checks distinctly from fresh passes. Retain
finite/missing-gradient checks during updates and historical failed receipts.
The explicit existing online B200 target is an infrastructure exception to the
Slurm default: the matched recipe has measured peaks around 147 GiB, above the
local 96 GB GPU capacity. Do not substitute a smaller-memory recipe.

Evaluate only the frozen canonical 3,012-row CoT-removed ID suite (STRIDE and
Gloom), preserving visible prose and source-grouped holdouts. Verify checksums
and original/transformed trajectory lineage disjointness. The six-source OOD
suite is outside preparation, evaluation and selection. Evaluate final masters
with a bounded training-source eager canary and require every serving adapter
and base to have correlation >=0.99 and mean score difference <=0.02, plus a
nonzero adapter effect, before its full ID evaluation. Use one persistent vLLM
engine for all candidates, constrained one-token decisions and explicit decision
logprobs; record ranking, calibration, threshold diagnostics and score ties by
source. Prediction caches bind config, instruction, serving artifact and input.

Selection: replace the fresh soft-only control only for macro normalized
pAUROC@20 gain >=0.005, no source loss >0.01, and macro Brier regression <=0.005.
Rank eligible candidates by macro pAUROC@20, then AUROC, then lower Brier.
This is a one-seed screen; confirm a winner across seeds before promotion or
final OOD evaluation. Freeze the final epoch checkpoint; do not select epochs.
Stop on hash/recipe drift, overlap, missing targets, recipe mismatch or failed serving
gates, OOM, nonfinite values, truncation or incomplete evaluation. Preserve
failures; never adapt the candidate grid from results.

Prepare and run on the existing authorized B200:

```bash
.venv/bin/python -m experiments.monitoring_hard_labels.prepare
.venv/bin/python -m experiments.monitoring_hard_labels.run
```

Artifacts: `data/monitoring_hard_labels/`, `results/monitoring_hard_labels/`.
Logs: `logs/runpod/monitoring_hard_labels/`. The serial runner trains all four,
executes bounded master reference canaries in the pinned training environment,
then serves all four in the clean vLLM environment and writes `summary.json`.
It does not publish models or change Pod lifecycle. Agent monitoring depends on
in-chat scheduling support; process persistence is not an agent heartbeat.

On 2026-10-02 the user explicitly requested skipping repeated validation for an
already validated recipe, then made this a standing preference. Version 2
reuses `results/student_injection_awareness/4b/regular/causal_adapter/training_metadata.json`.
The version-1 startup was stopped before optimizer updates; all diagnostic
receipts and logs remain under `startup_diagnostics_before_user_skip/`.
The compiled training policy and runtime finite-gradient checks remain active.

For an evaluation-only recovery after all training/reference receipts exist, use
`python -m experiments.monitoring_hard_labels.run --resume-evaluation`.
This preserves completed training and reference scores; serving still verifies
their adapter/prompt identities and serving parity. The launcher explicitly adds
the virtualenv and CUDA executable directories to `PATH` for FlashInfer JIT.

Completed 2026-10-03 (Amsterdam time): all four candidates trained 272 updates
and scored all 3,012 canonical ID rows. The frozen rule selects hard fraction
0.3 (soft fraction 0.7): macro pAUROC@20 0.888486 versus 0.860331 for soft-only,
AUROC 0.962873 versus 0.954951, and Brier 0.071914 versus 0.081820. Both ID
sources improve. Hard-only regresses macro ranking/calibration, driven by Gloom.
This remains a one-seed ID screen, with no OOD evaluation or promotion. See
[the completed finding](../../docs/findings/monitoring_hard_labels.md) and
`results/monitoring_hard_labels/summary.json` for the complete comparison.

OOD follow-up, explicitly requested 2026-10-03: freeze the ID-selected final
`hard030` checkpoint and its matched `hard000` control at LR 2e-5. Score the
unchanged strict 6,395-row, six-source suite in one persistent vLLM engine,
candidate first. The older regular OOD baseline used LR 5e-5, so it cannot
isolate the hard-label intervention. This follow-up reports transfer; it does
not select another fraction/checkpoint, promote a model or alter the ID rule.
The user's request authorizes this OOD evaluation before multi-seed confirmation.

`ood_config.yaml` freezes source/manifest hashes and each adapter's completed
parity/artifact/reference/ID-result identities. Reuse those successful parity
receipts only with unchanged model, runtime versions, GPU and GDN backend;
record `performed_this_run=false` in each OOD parity-reuse receipt. Stop on
identity/runtime drift, truncation, nonfinite scores or incomplete membership.
Preserve the original ID config, data manifest, receipts and summary. Prepare
locally, transfer the separate OOD manifest, and install the byte-identical
already cached regular OOD inputs on the B200:

```bash
.venv/bin/python -m experiments.monitoring_hard_labels.evaluate_ood --prepare
.venv/bin/python -m experiments.monitoring_hard_labels.evaluate_ood \
  --install-cached-input data/student_injection_awareness/regular/ood.jsonl
.venv/bin/python -m experiments.monitoring_hard_labels.evaluate_ood
```

The independent runner uses `ood.campaign.lock`, `ood_status.json`, resumable
`4b/{hard030,hard000}/ood/` predictions, and `ood_summary.json` with membership
and metric recomputation. Redirect its output to
`logs/runpod/monitoring_hard_labels/ood.log`. Agent follow-ups require an
available in-chat scheduler; no such tool is available in this session.
