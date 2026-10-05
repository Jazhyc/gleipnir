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

The current worker is PID `11905` on the existing US-NC-2 B200, session
`resident02`. Durable control/status files live at
`results/b200_mlp_gemm/resident02/`, with the active log at
`logs/runpod/b200_mlp_gemm/resident02/worker.log`. Inspect status and process/GPU
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

The initial queue completes successfully. Baseline means are 3.65854/3.67101
seconds per update, with identical losses, gradients, physical partitions and
final FP32 adapters after reset. All sixty actual updates add zero compiler
graphs, plans or specializations. The repeat takes 83.44038 seconds for twenty
updates without preparation. The worker is left idle with model/caches resident.

Shape/context attribution assigns 9.67% of summed GPU kernel time to frozen GDN
projections and 6.62% to LoRA GEMMs; GDN scan/convolution/normalization accounts
for another 25.24%. The next practical GEMM candidate is NVFP4 for large frozen
GDN QKV/Z/output projections, with shared input packing where feasible. Preserve
BF16 recurrence and small gate projections plus FP32 gates/normalization. This
candidate is proposed, not implemented or numerically accepted. The findings
record exact receipts, scope and source provenance.

The first GDN candidate in `resident01` fails before preparation or optimizer
updates because its installer assumes an ordinary bound forward rather than the
disabled Transformers wrapper. The original worker exits on that exception.
Preserve its failed request, source and logs. The corrected integration preserves
the Transformers Accelerate hook and disabled forward boundary. The replacement
worker reuses the same GPU and network-volume caches; its initial queue omits the
already-completed profiling run. Scoped candidate failures now restore baseline
state and return to idle when the CUDA context remains usable. Damaged CUDA
contexts and failed baseline reset checks still stop the worker.
