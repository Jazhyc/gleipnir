# Tool-trajectory hard-label strength screen

Status: prepared and submitted on the existing authorized B200, 2026-10-02.
Quality results are pending; this record does not change the selected objective.

Completed tool-trajectory students previously used Kimi K3 soft targets only.
Earlier hard-label anchoring experiments were in the deception domain. The
historical soft-only scaling restriction remains a reproduction contract; this
new user-authorized monitoring screen explicitly introduces source hard labels.

Freeze four normalized hard fractions 0, 0.1, 0.3 and 1 at LR 2e-5. Use the
regular compact student prompt, 8,688 monitoring rows, one epoch/272 updates,
seed 0, rank 128/alpha 256 and a shared initial FP32 adapter. Preserve the
selected packed BF16 B200 recipe and every isolation, memory, finite-gradient
and serving gate. Source labels are separate from unchanged cached soft targets.
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
throughput or a campaign ETA. Training and the queued ID evaluation remain
in progress; no quality result is available yet. The collected current kernel
receipt explicitly records `performed_this_run=false` and the reference SHA256
`91fadffbe9e69523c1c65cf485393ce2cc2f08aea8dafb2987f7c7663fa5f69f`,
with no fresh `passed` or `finite` field.
