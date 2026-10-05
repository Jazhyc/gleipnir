# Resident B200 FP4 optimization baseline

On 2026-10-05 the user selected combined native FP4 MLPs with BF16 FA4 as the
baseline for further training-stack timing improvements and requested keeping
a training worker resident. The subsequent user message explicitly extends
that persistence preference to future sessions; it is recorded in `AGENTS.md`.

This selection changes the systems-comparison reference, not the general
quality-validated training profile. Retain FP32 master adapters and historical
failed strict FP4 loss/gradient receipts alongside explicit timing acceptance.
The existing BF16 FA4 default remains available. See the
[MLP findings](../findings/b200_mlp_gemm.md) for warmed speed and numerical limits.

Keep the model, compiled modules, native plans and packed frozen weights alive
across compatible trials. Each matched trial resets the original FP32 masters,
AdamW/scheduler state, RNG and data order. Establish reset correctness with two
baseline trajectories once per worker, then reuse those receipts. Keep finite/
missing-gradient checks during actual updates and targeted checks for changed
arithmetic. Prepare benchmark shapes once; actual-update compiler/plan counters
verify convergence. Ordinary optimization trials do not require a second full
preparation replay or unchanged global diagnostic gates.

The current worker is PID `10916` on the existing US-NC-2 B200, session
`resident01`. Durable control/status files live at
`results/b200_mlp_gemm/resident01/`, with the active log at
`logs/runpod/b200_mlp_gemm/resident01/worker.log`. Inspect status and process/GPU
health before reusing it in a future session; recorded PID alone does not prove
the worker remains alive. No new billable capacity is implied by this preference.

The control command is `experiments.b200_mlp_gemm.resident_launch`. Baseline and
GEMM-shape diagnostic requests are bounded to twenty updates. Candidate requests
load the fixed experiment-owned `resident_candidate.py` file using a recorded
source checksum; each trial archives the exact source and targeted validation
receipt. The candidate's context manager restores the baseline afterward.
Compatible changes can therefore be submitted without restarting Python.
Submission/status commands do not import the training stack. Kernel/runtime,
parameter-layout or unsupported-envelope changes may require a new worker and
targeted diagnostics; retain network-volume compiler caches across that restart.

The initial queue is two uninstrumented FP4 baseline trajectories followed by
one shape-recording GEMM trace at update 15. Its purpose is to distinguish frozen
GDN/full-attention projection work from LoRA contraction and preserve ambiguous
shape matches. Select the next intervention from measured component cost and
complete-update timing against this resident FP4 baseline. The older BF16
control need not be repeated for each follow-up. No held-out quality promotion
or claim of BF16 numerical equivalence follows from this campaign.
