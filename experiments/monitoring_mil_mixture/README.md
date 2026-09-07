# Three-epoch MIL with full deception data

Status: the promoted two-GPU run and final ID evaluation completed; completion
audited on 2026-09-08.

The final run completed 1,398 updates. On all 3,012 historical ID rows, macro
pAUROC@20 fell from the monitoring-only MIL baseline's `0.895951` to `0.876761`;
AUROC was `0.957816` and Brier `0.077444`. Gloom pAUROC fell by `0.045623`, while
STRIDE increased by `0.007242`. This fails the exploratory gate. The comparison
changes data mixture, update count, schedule and execution, so it does not
isolate a causal effect of deception data. No OOD evaluation of this mixed
adapter was run. Full diagnostics are in
`results/monitoring_mil_mixture_ddp/summary.json`; the final FP32 master and
predictions are retained in the
[campaign inventory](../../docs/findings/gleipnir_improvement_shutdown_inventory.md).

Hypothesis: full-boundary deception distillation adds complementary supervision
to the selected 20%-monitoring, three-epoch MIL recipe. User requested one fresh
seed-0 run, not a continuation of an existing adapter. Retain the exact 1,738
monitoring parents and add all 13,149 historical deception examples, with their
original Kimi soft targets and student prompts. No new annotation or prefix loss.

Monitoring loss is `BCE(Kimi_full) + .25 * BCE(MIL_bag)`, with log-mean-exp
pooling, temperature 1 and at most eight endpoints. Deception loss is only
`BCE(Kimi_full)`. An explicit boolean `mil_enabled` field is true only on
monitoring rows. An empty MIL bag yields zero auxiliary loss, not a fabricated
action or a second full-boundary target. Average auxiliary contributions over
all parents in the batch, not just MIL-eligible parents. Default unmarked rows
retain the historical MIL behavior; record enabled/disabled counts in metadata.

Reuse the selected three-epoch MIL systems recipe: pinned Qwen3.5-4B revision,
LR2e-5, linear decay/3% warmup, AdamW, zero weight decay/dropout, rank128/alpha256
NF4 double quantization/BF16 compute, FP32 master adapters, microbatch1/accum32,
all-layer checkpointing, linear-shell compilation, required flash SDPA and
pinned FLA0.5.2/causal-conv1d kernels. The initial one-GPU run was stopped
before training steps at the user's request to test two-GPU DDP. Preserve
global batch32 using microbatch1/accum16 on each of two existing H100s.

The bounded `ddp-screen` phase compares eight updates on the same deterministic
256-parent mixture (30 monitoring, 226 deception), plus a separate longest-row
DDP preflight. Both throughput arms use non-reentrant activation checkpointing.
MIL selected-token projection runs through DDP.forward, never around its
reducer. CPU two-process tests compare accumulated gradients with an unsharded
reference, including a short final window; actual training requires exactly
matching trainable replicas before accepting completion. Retain original frozen
one-GPU artifacts and record new code hashes in a separate screen manifest.
Require finite loss, original kernels/precision, and >1% measured steady speedup
before promotion. Standard even-batch sharding may repeat one of 14,887 parents
per epoch; record that if promoted. No additional scientific objective arms.

Three epochs cover all 14,887 rows (1,398 updates versus 165 in the monitoring-
only baseline). No source reweighting or oversampling. This preserves three
monitoring exposures per parent but changes data diversity, update count and
schedule context; it is not a matched-compute causal attribution to deception.

Before full training, use one accumulation-32 step interleaving the 16 longest
monitoring rows by existing Qwen token count with the 16 longest deception rows
by UTF-8 byte length. This exercises both eligible and disabled MIL paths and
the largest monitoring context at rank128, plus a compile canary. Require finite
loss, correct 16/16 eligibility counts, pinned kernels, precision and batch.
Retain the frozen original mixed-artifact checksums and aligned target audit;
verify the exact selected monitoring prompts/labels/trajectory provenance and
the existing ID separation before launch. Never copy data into tracked files.

Evaluate only the final model on all 3,012 unchanged ID examples, after FP32
master/export serving parity. Baseline is three-epoch monitoring-only MIL:
macro pAUROC .89595082, AUROC .96692054, Brier .07260045. Apply the existing
exploratory +.005 pAUROC/no >.01 source loss/no >.005 Brier regression gate.
Report ranking, calibration, threshold diagnostics and ties by source. No OOD,
intermediate endpoint selection or automatic follow-up arms. This remains
single-seed adaptive ID development, not confirmed generalization.

Stop on provenance drift, invalid eligibility, preflight or training failure,
nonfinite loss, serving mismatch or incomplete evaluation. Evaluation hang
recovery uses the shared bounded watchdog; training is not automatically retried.
No timed agent scheduler is available in this session; the executable pipeline
is not an agent heartbeat.

### Distributed startup findings

The first two DDP attempts failed the existing compiled/eager logit gate on the
2,048-token preflight, before training. Both ranks reproduced eager logits
`[7.65625, 9.0625]` versus compiled `[7.8125, 9.1875]`; isolated rank-local
compiler caches did not fix it. Tolerances were not widened. Attempt 3 tests
explicitly eager DDP, retaining FLA and causal-conv1d, against compiled single-GPU
training. With compilation policy `none`, the same-weight check is an eager
repeatability check, not compilation parity.

Attempts 1/2 also revealed a screen-design issue: membership manifests retain
source order, so merely interleaving their keys did not interleave training.
Their timing is not used for promotion. Attempt 3 explicitly materializes the
same 256 parents in stable mixed order, and the longest preflight in alternating
monitoring/deception order. Full-campaign data and historical runs are unchanged.
An append-only source-amendment record documents enabling the eager canary
before the attempt-3 DDP child launch; compiled control behavior was unchanged.

Focused tests cover zero MIL contribution from disabled rows, actual Gloo
gradient reduction, Trainer/Accelerate full and partial accumulation windows,
launcher arguments, and deliberate replica-divergence rejection.

Attempt4 passed after preserving the FP32 projection and seeding existing Triton
cache entries before isolating rank-local writes. The accepted eight-step
comparison is 16.5840 versus 10.4542 seconds/update (steps3–8): 1.5863x throughput.
The fresh two-GPU full run lives under `results/monitoring_mil_mixture_ddp/` and
uses eager shells, unchanged fast kernels, accumulation16/rank and globalbatch32.
See [the systems finding](../../docs/findings/monitoring_mil_ddp.md) for failed
attempts, precision tests, cache effects, sampler padding and interpretation.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.monitoring_mil_mixture.run promote-ddp \
  --screen results/monitoring_mil_mixture/ddp_screen_attempt4 \
  --result-dir results/monitoring_mil_mixture_ddp
PYTHONPATH=src .venv/bin/python -m experiments.monitoring_mil_mixture.run run \
  --result-dir results/monitoring_mil_mixture_ddp
```

```bash
PYTHONPATH=src .venv/bin/python -m experiments.monitoring_mil_mixture.run prepare
PYTHONPATH=src .venv/bin/python -m experiments.monitoring_mil_mixture.run run
```
