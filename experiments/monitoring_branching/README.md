# Gradient-preserving prompted branches

Hypothesis: sharing the causal trajectory computation across all intermediate
decision suffixes can reproduce independent-prefix training while reducing
repeated forward/backward work. This is a systems change, not an auxiliary
classifier: every branch retains the exact existing prompted decision interface.

Baseline: independently forward each complete tokenized prefix prompt plus the
full endpoint; use the same normalized soft BCE losses and targets. Intervention:
process shared trajectory segments once, fork differentiable KV/conv/recurrent
states at boundaries, and process each original suffix independently. Never feed
a branch's closing tags or prediction tokens into the continuing trajectory.

Token sharing uses actual longest common token prefixes, not character offsets;
BPE joins can change the last trajectory token when the suffix is appended.
Avoid one-token cached continuations because they invoke inference-specific
in-place convolution/recurrent kernels. Move a split earlier when necessary,
without changing any request's token sequence.

First gate: tiny mixed full/linear-attention Qwen CPU model in deterministic FP32,
eager attention, no dropout, no checkpointing. Compare decision logits, the
normalized objective, and every parameter gradient against independent requests.
Assert branch-state isolation and fewer processed tokens. This reference check
intentionally uses the Torch fallback and is not a production training recipe.

Second gate (pending idle GPU capacity): real pinned FLA/causal-conv kernels,
BF16 and rank-128 QLoRA; compare scores, gradients and a matched optimizer update
before timing representative and longest trajectories. Require finite gradients,
nonzero gradient through shared history, preserved full Kimi targets, no leakage,
and matched parent normalization. No ID-based tuning of systems tolerances.
Do not switch a running campaign or claim throughput gains before this gate.

The prototype fails closed on checkpointing and dropout. Differentiable caches
retain activations: all-boundary memory may exceed the current sampled-prefix
recipe. Functional checkpointing or a layerwise implementation may be needed;
ordinary inference-cache reuse or detached states is not an exact substitute.

Run the bounded CPU correctness entrypoint:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m experiments.monitoring_branching.run
```

Current prefix-weight runs are untouched. No GPU benchmark is automatically
queued, and no monitoring of those runs is resumed by this experiment.

## Initial CPU result

The mixed-attention FP32 canary passed on 2026-09-06: maximum decision-logit
error 1.49e-8 and maximum parameter-gradient error 2.38e-7. The four exact
requests processed 40 token positions with sharing versus 75 independently.
This is correctness evidence for the eager reference path only, not a measured
H100 speedup or proof that FLA final-state gradients and QLoRA match. Full-scale
memory optimization, production-kernel parity, and trainer integration remain
pending. Do not replace the active sampled-prefix runs with this prototype.

## Authorized all-prefix experiment

User now requests one all-prefix run, weight 0.1, using both GPUs for training.
Preserve LR 2e-5, one epoch, seed 0, all 8,688 parents and original Kimi
endpoint targets. For each eligible parent use
`(L_full + 0.1 * sum(L_prefix)/K)/1.1`; prefix-free parents retain L_full.
Include all 133,947 approved cached prefix boundaries, retaining the previously
authorized numerical-cache exception; do not relabel or rescale targets.
Compare final ID evaluation with full-only and one-sampled-prefix weight 0.1;
no OOD selection or automatic extra hyperparameter search. Do not claim an
all-prefix benefit based on numerical-kernel canaries.

Before trainer integration, run a bounded real 4B QLoRA parity diagnostic on
GPU 0 using the existing trained 0.1 adapter to exercise nonzero A/B gradients.
Use identical synthetic requests of length 259/516/1027, production FLA and
causal-conv1d, BF16, eager SDPA and zero dropout. Freeze maximum margin error
0.1, probability error 0.02, global gradient relative L2 0.05 and cosine 0.995.
Stop on missing/nonfinite gradients or a failed gate. This short gate is not
sufficient to launch training: optimizer-step, long-sequence memory, functional
checkpointing and two-GPU gradient/normalization checks remain required.

## Production preflight progress (not a training launch)

Added on-demand exact all-prefix request reconstruction in
`gleipnir.branch_data.BranchDataset`, preserving cache/parent/target checksums,
and opt-in functional segment checkpointing using fresh cache containers during
both forward and recomputation. The maximum prefix count is 99 (median 17 among
6,348 eligible parents); all 133,947 targets remain in scope. GPU canaries load
the trained weight-0.1 adapter solely to exercise nonzero A/B gradients; an
eventual matched full training run must initialize fresh adapters from the base.

The short synthetic real-4B production gate passed with identical scores,
gradient relative L2 0.017312, cosine 0.999850. Checkpointing reproduced that
pass (relative L2 0.017320). But real parent 0 failed: margin error 0.125,
probability error 0.001445, gradient relative L2 0.246704, cosine 0.992256.
Alignment to 64-token boundaries preserved exact requests but still failed:
relative L2 0.225525, cosine 0.996498. Do not relax the frozen gates or call
small probability error evidence of correct training gradients.

Longest-token parent 8142 also has 99 prefixes. Single-GPU checkpointed execution
ran out of memory. Explicit two-GPU layer placement (layers 0–15/GPU0,
16–31/GPU1, tied input/output embeddings both GPU0) completed a bounded
forward/backward/AdamW update in 123.28 seconds. Peak allocations were
67,653,889,024 and 64,458,370,048 bytes. This establishes one-case memory
feasibility, not full-workload correctness or a mean training ETA. Data
parallelism would replicate the overflowing per-trajectory state, so the
proposed two-GPU training arrangement is model parallel, with accumulation 32.

FP32 selected-token head projection on the two-GPU aligned path reduced real
parent-0 gradient relative L2 to 0.081498, still failing 0.05. A further optional
variant computes the full Kimi endpoint independently and shares only the prefix
trunk. Its first GPU check exposed an embedding-gradient hook left behind by
Transformers' public checkpoint-enable method. The implementation now toggles
layer checkpoint flags without installing that hook; a frozen-embedding CPU
regression covers the issue. Eighteen focused CPU tests pass across checkpoint,
FP32-head, independent-endpoint, cache isolation, exact requests and prefix loss.
The corrected hybrid GPU gate remains pending; no all-prefix run is launched.

Diagnostics and failures are retained under `results/monitoring_branching/` and
`logs/lambda/monitoring_branching/`. Optional precision/alignment/endpoint variants
are systems diagnostics, not teacher-target calibration. No annotation changes,
threshold loosening, OOD tuning, or long-run launch follows from a failed gate.

The corrected hybrid (independent full endpoint, shared aligned prefixes,
FP32 two-token projection, two-GPU layer split) passes real parent 0: margin
error 0.015581, probability error 0.000434, gradient relative L2 0.014387 and
cosine 0.999897. Next freeze additional positive fixtures by first eligible
parent per source with 2–10 prefixes and <=4,096 tokens: 487 (BashArena),
222 (BashBench), 2608 (Gloom), 132 (STRIDE), plus repeat parent 0. Retain
all previous numerical gates; additionally require first-step Adam adaptive
update relative L2 <=0.20, computed from identical zero-moment initial states.
This diagnoses sensitivity near zero gradients, not full optimizer-state parity.
Then recheck memory/finite updates on 8142 (99 prefixes, longest tokens) and
7706 (longest-character parent, nine prefixes and larger segment gaps), using
the exact corrected hybrid variant. No training launch on any failed gate.

## Broader gate stopped: full run remains unlaunched

The frozen `preflight.yaml` series (supervisor 180987) stopped at parent 222;
no Gloom/positive-STRIDE checks or hybrid memory checks ran after that failure.
Results/logs are preserved locally and remotely; both GPUs are idle.

| Parent/source | Probability max error | Gradient relative L2 | Gradient cosine | First Adam adaptive-update relative L2 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 / STRIDE | 0.000434 | 0.014730 | 0.999892 | 0.078272 | Pass |
| 487 / BashArena | 0.003814 | 0.010206 | 0.999948 | 0.074579 | Pass |
| 222 / BashBench | 0.000378 | 0.072486 | 0.997390 | 0.160924 | Fail |

BashBench fails only the unchanged 0.05 relative-gradient limit. Its reference
gradient norm is small (0.042600; absolute gradient L2 difference about 0.003088),
so relative error is sensitive to small numerical changes near a low-loss
solution. This is not proof of broken autograd or a harmful training update.
Conversely, small probability error does not establish matched gradients.
Keep the failure visible; do not silently introduce a looser tolerance after
seeing it. The CPU reference remains correct, but production numerical
equivalence is not established across this bounded screen.

An approximate-training exception would require explicit user agreement before
finishing the remaining gates/integration and launching. The demonstrated
two-GPU 99-prefix memory pass belongs to the fully shared variant, not yet the
corrected independent-endpoint variant. No full-run ETA is established.

## Explicit exploratory authorization and remaining checks

On 2026-09-06 the user approved proceeding: "Interesting. It should be fine.
This is an exploratory experiment." `training.yaml` pins the original failed
BashBench diagnostic SHA and preserves the earlier cache-noise exception.
This authorizes approximate numerical branching, not target changes, detached
history, missing/nonfinite gradients, memory failures, or silent gate relabeling.
The original failed series and tolerances remain unchanged as historical results.

The remaining frozen Gloom and STRIDE fixtures pass the original limits:
gradient relative L2 0.015113 and 0.011901; cosine 0.999886 and 0.999929.
The **exact hybrid variant** now passes both memory/finite-update checks:
parent 8142 (99 prefixes) takes 126.24 seconds and peaks at 76.686/69.648 GB
allocated; parent 7706 takes 17.88 seconds and peaks at 35.618/31.922 GB.
These include AdamW with zero weight decay and finite post-update masters.
Memory headroom is tight; these extremes are not mean-workload ETA estimates.

The training integration lives in reusable `gleipnir.branch_model` and
`gleipnir.branch_trainer`, with one config-driven experiment entrypoint:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.monitoring_branching.train prepare
# Use the isolated, verified production-kernel environment for the GPU phases:
PYTHONPATH=src .venv/bin/python -m experiments.monitoring_branching.train smoke
PYTHONPATH=src .venv/bin/python -m experiments.monitoring_branching.train pipeline
```

Freeze one epoch, seed 0, LR 2e-5, fresh rank-128/alpha-256 QLoRA masters,
NF4 double quantization/BF16 compute, zero dropout/weight decay, ordinary AdamW,
linear schedule with 3% warmup, norm clipping 1.0. Share only prefix history,
64-token-aligned exact branches, FP32 selected-token projection, functional
segment checkpoints and ordinary full-endpoint layer checkpoints. Compilation
is disabled for this new cache-aware path; no throughput equivalence with the
previous compiled sampled-prefix loop is claimed. Two-GPU **layer** parallelism,
microbatch one and accumulation 32 give equal parent mass. The final 16-parent
window normalizes by 16. Shuffle all parents once with a dedicated seed-0 Torch
generator; this is deterministic but not asserted identical to HF Trainer order.

The eight-parent fresh-adapter integration smoke uses the same first shuffled
parents, objective and accumulation, with warmup disabled solely to exercise a
nonzero optimizer update. Smoke weights are discarded. Full training restarts
from fresh adapters, visits all 8,688 parents and 133,947 prefixes, saves resumable
adapter/optimizer/scheduler/RNG checkpoints every 32 updates and at the endpoint,
and checks finite gradients/masters at every update. No classifier head is added.
Original Kimi endpoint targets and all Qwen cached prefix targets are unchanged.

The pipeline releases training memory before existing causal-master/vLLM serving
parity, then evaluates the one final adapter on all 3,012 ID rows in a persistent
vLLM engine. Compare with full-only (0.871066 pAUROC) and sampled-prefix w0.1
(0.876688); keep the prior exploratory +0.005/no >0.01 source loss/no >0.005
Brier regression rule versus full-only. No intermediate ID selection or OOD use.

The fresh-adapter integration smoke passed: eight parents, 162 prefixes, 211.39
seconds, finite gradient norm 30.17 before clipping and a nonzero AdamW update
at LR 2e-5. Peak allocations were 51.840/45.701 GB. All 169,869,312 trainable
parameters are FP32 masters. Twenty-seven focused tests pass, including exact
optimizer-checkpoint roundtrip. The frozen ID descriptor validates all 3,012 rows.
Full pipeline supervisor 183692 was launched on the reserved two-H100 node;
training restarts from fresh adapters and does not reuse smoke weights.
Eight-parent timing suggests roughly 2–3 days, not a settled workload ETA.
This session has no agent scheduling tool; startup is checked actively, but no
automatic ten-minute agent follow-ups are claimed after the chat turn ends.

## Replacement: one fixed-compute endpoint, not a full epoch

The user cancelled the multi-day run and its queued evaluation, then explicitly
restricted the replacement to **1x only**. Supervisor 183692 and child 183972 were
identity-checked and their process group stopped; both H100s were confirmed idle.
One optimizer update (32 parents/673 prefixes, 860.43 seconds) had completed;
additional partial-window work was discarded. Logs/progress/cancellation records
are preserved, and no model or ID result from this aborted run is promoted.

Hypothesis: all-prefix branching improves ID monitoring at the same nominal
allocated GPU-time as the existing one-sampled-prefix weight-0.1 recipe. The
reference is the already evaluated seed-0 sampled model, ID macro pAUROC
0.876687996, not a newly selected checkpoint. Its frozen training metadata records
13,065.9032 seconds on one H100: **3.62942 GPU-hours**. Therefore `compute1x.yaml`
allows **6,532.9516 seconds (1h 48m 53s)** on two H100s. Reuse the historical
reference rather than spend its training budget a second time. There is no 2x arm.

Intervention: fresh base/rank-128 adapter, weight 0.1, same exact shared-prefix
objective, kernels, two-device placement, and seed-0 shuffled parent order.
Process every prefix for each visited parent, but stop by elapsed compute instead
of requiring all 8,688 parents. This tests compute efficiency: it necessarily
trades broader trajectory coverage for denser supervision per trajectory.
It does not isolate supervision density at matched data exposure, nor establish
what a fully converged all-prefix epoch could achieve. The earlier numerical
exception remains explicit, and no additional teacher calls are made.

Count wall time inside the training loop times two GPUs, including tokenization,
communication, optimizer work and periodic checkpoint overhead. Setup, final
export, inference and preflight cost are separate from this training comparison;
the cancelled attempt is a sunk experimental cost, not hidden in the 1x budget.
No new parent begins after the deadline. Finish the in-flight parent, rescale a
partial accumulation window to its actual parent count before clipping, update,
and save. Report actual GPU-seconds and any one-parent/save overrun, not an exact
FLOP match. Budgeted resume is disabled because discarded work needs explicit
aggregate accounting. All earlier failure/provenance/finite-gradient gates remain.

Use peak LR 2e-5 with the same 3% warmup and linear decay **over the short budget**,
evaluated at the elapsed-time midpoint of each optimizer window. Do not simply
cut off the old 272-update schedule during warmup. This scheduler convention is
frozen before ID evaluation; it is not an extra LR search. Save every four updates
and at the budget endpoint. Tests cover deadline crossing, no further parent
execution, correct partial-window gradients, a nonzero first update and reporting
of consumed two-GPU time. The previous eight-parent fresh-adapter GPU smoke is
reused only after checking the numerical-path source hashes are unchanged.

One final serving-parity-gated ID evaluation follows, with no intermediate ID
selection or OOD use. Compare ranking and per-source/calibration diagnostics
against sampled w0.1; retain the exploratory +0.005/no >0.01 source loss/no >0.005
Brier regression thresholds, now relative to that sampled reference. Report the
full-only model separately, and retain negative results.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.monitoring_branching.train prepare --config-name compute1x
PYTHONPATH=src .venv/bin/python -m experiments.monitoring_branching.train pipeline --root results/monitoring_branching_compute1x
```

The single 1x pipeline was launched as supervisor 184536. Thirty focused tests
pass, including a simulated deadline crossing during a partial optimizer window.
The old full-epoch evaluation is cancelled; only the replacement endpoint is
queued. Automatic agent heartbeats remain unavailable in this session.
