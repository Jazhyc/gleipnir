# Tool-trajectory hard-label strength screen

Status: all four training runs and canonical ID evaluations completed,
2026-10-03 (Amsterdam time). The frozen one-seed rule selects hard fraction 0.3.
This is an ID screen result; seed confirmation and OOD evaluation remain pending.

Completed tool-trajectory students previously used Kimi K3 soft targets only.
Earlier hard-label anchoring experiments were in the deception domain. The
historical soft-only scaling restriction remains a reproduction contract; this
new user-authorized monitoring screen explicitly introduces source hard labels.

Freeze four normalized hard fractions 0, 0.1, 0.3 and 1 at LR 2e-5. Use the
regular compact student prompt, 8,688 monitoring rows, one epoch/272 updates,
seed 0, rank 128/alpha 256 and a shared initial FP32 adapter. Preserve the
selected packed BF16 B200 recipe. Version 2 reuses prior startup validation as
documented below, retaining finite-gradient and serving checks. Source labels
are separate from unchanged cached soft targets.
The hard-only launch omits the teacher artifact from the trainer configuration
while preserving the cache and its checksum in campaign provenance.

The canonical CoT-removed ID suite contains 3,012 rows; both its input and
manifest hashes are frozen. Preparation verifies source/target identity coverage,
original and transformed trajectory lineage disjointness, prompt hashes and the
resolved systems profile. No OOD inputs are prepared or evaluated.

Use a fresh soft-only control under the same recipe and LR. Replace it only for
macro normalized pAUROC@20 gain >=0.005, no per-source loss >0.01 and macro Brier
regression <=0.005, breaking ties by AUROC then lower Brier. Require seed
confirmation before promotion. Full source ranking, calibration, thresholds and
ties remain in per-condition ID reports.

The existing B200 was read back RUNNING and probed idle with 183,359 MiB VRAM.
Its matched historical peak around 147 GiB exceeds the local 96 GB GPU capacity,
so the explicit B200 target is the documented exception to the Slurm default.
No new billable capacity or Pod lifecycle change is requested.

Preparation passed locally and all transferred materialized files were verified
on B200. Relevant CPU checks passed: 74 packing/data/evaluation checks, 46
additional launcher checks, and 11 checks after the hard-only command fix.
Repository Ruff passed. The existing completed packed training receipt also
passed the new loss/recipe metadata audit. Reusable data, launch and evaluation
code now lives under `src/gleipnir/`; historical entrypoints preserve their APIs.

Artifacts: `data/monitoring_hard_labels/`, `results/monitoring_hard_labels/`;
logs: `logs/runpod/monitoring_hard_labels/`. A serial runner trains four cells,
executes bounded training-source master canaries, evaluates all cells in one
persistent vLLM engine and writes the frozen selection to `summary.json`.
It records stage failures and holds a lock against duplicate campaigns.
No agent scheduling capability exists in this session: startup can be inspected
in the active turn, but automatic agent follow-ups after the turn are unavailable.
Process persistence does not provide those follow-ups.

See [the experiment contract](../../experiments/monitoring_hard_labels/README.md).

## First-cell startup diagnostics

The fresh soft-only control's token audit covers 83,816,369 tokens across 8,688
examples, maximum 29,337 under the 29,696-token cap, with zero truncation.
The model loader verifies the BF16 frozen base and all 256 FP32 adapter tensors.
The bound original FLA/convolution modules match the pinned recipe.

Its FlashQLA-versus-original-FLA strict gradient comparison reports relative L2
0.5719857351, `passed=false`, `finite=true` and
`accepted_for_selected_recipe=true` under `selected_finite`. The previous
completed 4B regular campaign reports 0.5762394181 with the same failed-strict,
finite-accepted status. This is the recipe's existing numerical limitation;
neither receipt establishes original-FLA gradient equivalence. Preserve the
separate historical strict packing gates and longest-batch memory preflight.
Fresh compiler caches require cold forward/backward kernel builds at startup.

## User-selected validation reuse (version 2)

The user subsequently requested beginning training without repeating gates for
an already validated recipe, and explicitly made this a standing preference.
The original startup process group was stopped before optimizer updates; its
logs, kernel/packing receipts and execution contract are preserved under
`startup_diagnostics_before_user_skip/`. The warm compiler caches remain.
Version 2 uses the same four fractions, data, prompts, LR and initial weights.
It reuses the completed regular-4B training receipt, with its checksum, and marks
all reused comparisons/preflights as not performed in this run. Current metadata
does not claim fresh parity or finite diagnostic results. Kernel/source identity,
input provenance and missing/nonfinite-gradient checks during updates remain.
Adapter-specific master/serving score parity remains part of ID evaluation.

The completed zero-truncation audit is reused for byte-identical candidate inputs.
Normal training tokenization/model loading still occur. The restarted campaign
PID is 68393 on the same B200. The new reuse implementation and existing runtime
finite-gradient behavior passed 58 focused CPU checks. The standing instruction
is recorded in `AGENTS.md`; the historical profiles and failed receipts remain.

Version 2 completed its first optimizer update on 2026-10-02 at approximately
18:33 UTC, after 698 seconds of actual first-batch compilation/execution.
Runtime missing/nonfinite-gradient rejection remained enabled; the update
completed without a logged failure. The GPU held approximately 146 GiB and
showed active kernel execution. This cold first update does not establish steady
throughput or a campaign ETA. At this startup check, training and the queued ID
evaluation were in progress and no quality result was available. The kernel
receipt explicitly records `performed_this_run=false` and the reference SHA256
`91fadffbe9e69523c1c65cf485393ce2cc2f08aea8dafb2987f7c7663fa5f69f`,
with no fresh `passed` or `finite` field.

## Training completed and evaluation-only recovery

All four candidates completed 272 updates with the frozen loss weights, LR,
initial adapter and training population. Their training runtime receipts report
4,208.57 seconds (hard fraction 0), 3,424.86 (0.1), 3,431.71 (0.3) and 3,395.91
(1). Training and eager master-reference JSON receipts were collected locally.
ID results became available after the serving recovery below.

The first vLLM process failed during FlashInfer initialization because its
executable `PATH` omitted `.venv/bin`, hiding the already-installed `ninja`.
This is an evaluation launcher defect; no training checkpoint failed. Preserve
the original `campaign.missing_ninja.log`, `vllm.missing_ninja.log` and
`evaluation_failure_missing_ninja.json`. The launcher now explicitly includes
the virtualenv and CUDA binary directories. `--resume-evaluation` requires all
completed training/reference receipts and resumes only vLLM and summary, so it
does not repeat training or master references. Existing adapter/prompt identity
and serving checks remain mandatory. The recovery PID is 72968 on the same Pod;
no capacity or lifecycle operation was performed.

Six focused resume/contract checks and scoped Ruff passed. The resumed engine
loaded its cached graphs, completed FlashInfer warmup and passed soft-only
master/serving parity. At that startup check, full ID scoring was pending/in
progress and no quality selection was made. All four completed training metadata
receipts passed the frozen objective/recipe audit locally, using a checksum-
identical local reference path for the remote receipt.

## Completed canonical ID comparison

All four candidates scored the complete 3,012-row suite: 2,066 Gloom and 946
STRIDE examples, for 12,048 predictions. Each evaluation recorded 33,750,959
input tokens, maximum 29,513 under the 32,768 limit, with no truncation. The
measured scoring/metric loops took 659.60, 656.72, 656.74 and 655.48 seconds,
respectively (43.81 minutes total), excluding prompt preparation and engine
startup. The campaign reported complete; the serving engine shut down and the
B200 was verified idle with zero allocated GPU memory. The Pod remains online.

All metrics below are source macro averages on the same canonical ID population.
Hard fractions are normalized: `(1-alpha)*soft_BCE + alpha*hard_CE`.

| Hard fraction | pAUROC@20 | AUROC | Brier | Eligible under frozen rule |
| --- | ---: | ---: | ---: | --- |
| 0 | 0.860331 | 0.954951 | 0.081820 | Control |
| 0.1 | 0.868663 | 0.958013 | 0.079671 | Yes |
| **0.3** | **0.888486** | **0.962873** | **0.071914** | **Selected** |
| 1 | 0.811978 | 0.931630 | 0.123208 | No |

The 0.3 candidate gains 0.028154 macro pAUROC@20 (2.82 percentage points),
0.007922 AUROC, and lowers macro Brier by 0.009906. Both partial hard-label
mixtures meet the predeclared gain/source/calibration limits. No candidate grid,
checkpoint selection, threshold or eligibility rule was changed after scoring.

| Hard fraction | Gloom pAUROC@20 | STRIDE pAUROC@20 |
| --- | ---: | ---: |
| 0 | 0.783768 | 0.936894 |
| 0.1 | 0.793383 | 0.943942 |
| 0.3 | 0.816112 | 0.960860 |
| 1 | 0.645919 | 0.978038 |

The selected mixture improves Gloom by 3.23 points and STRIDE by 2.40 points.
Hard-only improves STRIDE by 4.11 points but loses 13.78 points on Gloom and
regresses macro calibration. This supports partial hard-label supervision with
the soft teacher distribution on this ID suite; it does not establish a general
hard-label advantage or select a deployment model across domains.

At threshold 0.5, selected source-macro recall is 0.875198 and FPR 0.055579,
versus 0.857215 and 0.060504 for the control. Selected pooled ECE is 0.037304
and log loss 0.270985, versus 0.039798 and 0.295860 for the control. Score ties
remain substantial: unique scores/tied rows are 1,655/1,357 (control),
1,612/1,400 (0.1), 1,729/1,283 (0.3), and 2,785/227 (hard-only). Full source
ranking, reliability bins and threshold diagnostics remain in each ID result.

All adapters passed serving parity. Adapter correlations were 0.999926,
0.999836, 0.999999 and approximately 1; mean absolute score differences were
0.001552, 0.003554, 0.000165 and 0.000008, with nonzero adapter effects.
The common base correlation was 0.999170 and mean difference 0.006410.

Final artifacts live under `results/monitoring_hard_labels/4b/`: FP32 masters,
serving copies, predictions, ID reports, parity and provenance receipts.
`summary.json` applies the frozen rule. Local coverage, raw-score,
ranking/calibration and selection verification is recorded in
`completion_audit.json`; `artifact_collection_audit.json` verifies checksums
and 256 FP32 tensors for every collected master and serving adapter. All eight
adapters, predictions, reports and logs were collected locally; duplicate
optimizer checkpoints remain on the Pod.
Logs preserve the numerical diagnostics and missing-ninja failure alongside
the successful recovery. No OOD or final-test result was used in ID selection,
no model was published, and promotion remains false. Confirm the 0.3 mixture across seeds
before promotion. Transfer evidence for the fixed candidate follows below.

## Fixed-candidate OOD follow-up (2026-10-03)

The user requested OOD evaluation of the ID-selected 0.3 adapter. Freeze that
final checkpoint and the screen's soft-only control at the same LR 2e-5 and
score all 6,395 strict OOD examples (AgentDojo, CUA SHADE Arena, IAC, MLE
sabotage, RogueEval and SHADE Arena). The previous regular OOD baseline used
5e-5 and cannot isolate the weight change. Evaluate the candidate first, then
its matched control, in one engine. Original ID selection remains unchanged;
OOD does not select a checkpoint or another weight. Multi-seed confirmation
and promotion remain outstanding.

The existing adapter-specific parity checks are reused with frozen artifact,
reference and ID-result checksums, unchanged vLLM/Torch/GPU/GDN identities and
explicit `performed_this_run=false` receipts. No new training or numerical
canary is requested. Source provenance and zero training-lineage overlap were
verified before launch; instructions and the strict OOD population are unchanged.
See `experiments/monitoring_hard_labels/ood_config.yaml`. Completed results are
recorded in `results/monitoring_hard_labels/ood_summary.json`.

Launched on the existing B200 Pod `alzfug70g5237b` at 2026-10-03 00:13:57 UTC
(02:13:57 Amsterdam), evaluator PID 74749. The independent launcher and receipt
are ignored artifacts under `results/monitoring_hard_labels/`. Startup was checked
during the launch turn. No in-chat scheduling capability is available, so recurring agent
follow-ups are not scheduled; detached process persistence is separate from
agent monitoring. The original completed ID status is preserved.

Completed and collected 2026-10-03: each adapter scored all 6,395 examples, for
12,790 predictions in one persistent engine. The frozen serving configuration
was vLLM 0.24.0, Torch 2.11.0+cu130, NVIDIA B200 and FlashInfer GDN prefill.
Both conditions reused their existing successful adapter-specific parity
receipts with `performed_this_run=false`; unchanged artifact/runtime identities
were verified. Each scored 55,346,879 prompt tokens, maximum 29,570, below the
32,768-token cap with zero truncation. Scoring took 998.50 seconds for the
candidate and 994.30 seconds for the control, excluding engine startup and the
initial rendering audit. The final campaign status is complete and the GPU is
idle; no recurring agent follow-ups were scheduled.

| OOD source-macro metric | Soft-only, LR 2e-5 | Hard fraction 0.3, LR 2e-5 | Candidate minus control |
| --- | ---: | ---: | ---: |
| normalized pAUROC@20 | 0.782860 | 0.788400 | +0.005540 |
| AUROC | 0.910372 | 0.912103 | +0.001731 |
| Brier (lower is better) | 0.140889 | 0.140996 | +0.000106 |

The macro pAUROC gain is 0.554 percentage points, substantially smaller than
the 2.815-point ID gain. Four of six sources improve in pAUROC, but the average
hides a large SHADE Arena regression:

| Source | Rows | Soft-only pAUROC@20 | 0.3 pAUROC@20 | Change, percentage points |
| --- | ---: | ---: | ---: | ---: |
| AgentDojo | 1,500 | 0.797942 | 0.875016 | +7.71 |
| CUA SHADE Arena | 955 | 0.777502 | 0.769860 | -0.76 |
| IAC | 1,443 | 0.847464 | 0.891054 | +4.36 |
| MLE sabotage | 148 | 0.997959 | 0.998980 | +0.10 |
| RogueEval | 1,499 | 0.955358 | 0.965127 | +0.98 |
| SHADE Arena | 850 | 0.320935 | 0.230365 | -9.06 |

SHADE Arena AUROC also falls from 0.669159 to 0.637178, and Brier worsens from
0.282530 to 0.316201. This corroborates a ranking and calibration regression
on that source, rather than merely a changed operating threshold.

At threshold 0.5, source-macro recall declines from 0.721728 to 0.705645 while
FPR declines from 0.100735 to 0.076263. Source-macro Brier is essentially
unchanged. Pooled ranking improves more than the equal-source mean: pAUROC@20
0.654394 to 0.694456 and AUROC 0.898207 to 0.913839; pooled Brier improves
0.147748 to 0.140698 and log loss 0.462060 to 0.449477. Pooled ECE worsens
0.098869 to 0.122032, with probability bias becoming more negative (-0.095456
to -0.122032). Calibration conclusions therefore depend on the metric and
population weighting. Unique scores/tied rows remain 2,121/4,274 (control) and
2,130/4,265 (candidate); full per-source diagnostics and reliability bins are
preserved in each report.

The matched control was useful: the historical soft-only LR 5e-5 adapter had
macro pAUROC@20 0.776238, AUROC 0.914348 and Brier 0.138607 on these same inputs.
It is a separate learning-rate condition. The hard-label effect reported above
uses the fresh LR 2e-5 control, with all other frozen training choices held fixed.

Adding 0.3 hard supervision gives a small OOD macro ranking gain
and stronger pooled ranking, with mixed calibration and substantial source
tradeoffs. The larger ID improvement does not establish a broad OOD advantage.
These are descriptive one-seed results; no uncertainty interval was calculated.
Keep the original ID selection intact. No OOD-based weight/checkpoint selection,
publication or promotion occurred; multi-seed confirmation remains outstanding.

All OOD predictions, reports, parity-reuse receipts and logs were collected
locally. `ood_completion_audit.json` verifies frozen membership/provenance,
finite raw logprobs, normalized-score reconstruction, zero truncation, complete
ranking and calibration recomputation, adapter identities, summary checksum and
preservation of the original ID summary. Its checked script is the ignored
`results/monitoring_hard_labels/audit_ood_completion.py`.
