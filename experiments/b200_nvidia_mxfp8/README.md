# NVIDIA MXFP8 full-attention training screen

Hypothesis: NVIDIA's Blackwell D256 MXFP8 causal GQA forward/backward engines
reduce complete Qwen3.5-4B LoRA update time relative to the user-selected BF16
FA4 recipe. Test eight full-attention layers; retain 24 BF16 FlashQLA GDN layers,
BF16 MLPs, FP32 rank-128 master adapters and the frozen systems-screen cohort.

The initial intervention calls dense BSHD MXFP8 separately for each example
within the existing packed physical row. It preserves sequence boundaries and
logical example weighting but sacrifices fused variable-length attention. All
conversion, scale-layout repacking, allocation and backward costs are included.
Pin cuDNN Frontend revision `51d9d06b574222378a3d806009accab098e73705` in an
isolated overlay; preserve the current environment and shared persistent caches.

Before model loading, check actual causal D256 GQA 16Q/4KV forward/backward
against independent FP32 attention, quantizer layout against NVIDIA's reference,
and cross-example isolation. Record within-sequence future-token quantization
effects separately. Forward relative L2 limit is 2%, gradient relative L2 5%.
Stop on unsupported engines, missing/nonfinite gradients, isolation failures,
native numerical gate failure, OOM or dependency incompatibility; preserve
failed receipts. The separate learning continuation below retains strict results.
No silent fallback or default promotion.

If native checks pass, perform fresh eager/compiled model packing checks and
longest-cohort memory preflight, then compare matched 20-update trajectories
from the same initial adapter, ten warmup and ten measured updates. Select no
checkpoint on held-out quality; this is a bounded systems screen, with no teacher
requests. Recommend further work only if complete measured updates are at least
5% faster; longer quality checks and replication remain necessary for promotion.

Artifacts: `results/b200_nvidia_mxfp8/`. Logs:
`logs/runpod/b200_nvidia_mxfp8/`. Startup and bounded execution are monitored in
the active turn; this session has no in-chat scheduling tool for later follow-ups.

```bash
bash experiments/b200_nvidia_mxfp8/bootstrap.sh
```

The bootstrap builds the pinned upstream snapshot from an ignored source archive
and resolves cuDNN 9.26 separately from the existing runtime. Exact installed
versions, archive checksum and effective cache paths must be recorded before
kernel execution. It does not modify `pyproject.toml` or `uv.lock`.

The first attempt executes MXFP8 forward, but cuDNN's graph validator rejects
one-token self-attention backward (`s_q = s_kv = 1`). Its failed receipt and
executed source are preserved at the artifact root. The experimental training
boundary handles that degenerate case analytically: output is V, Q/K gradients
are zero, and V gradients sum across grouped query heads. This explicit exact
BF16 case requires no matrix multiplication; sequences of two or more tokens
use MXFP8. `native02.yaml` continues with
supported lengths in `attempt02`, with unchanged numerical limits and caches:

```bash
source .cache-runtime.env
.venv/bin/python -m experiments.b200_nvidia_mxfp8.run \
  --config experiments/b200_nvidia_mxfp8/native02.yaml
```

The supported native suite executes both passes and all 24 fused-quantizer
payload/scale-layout checks pass. Against FP32, forward relative L2 is
3.11–4.55%, Q/K gradient relative L2 6.47–8.65%, and V gradients 3.56–4.84%.
Strict 2%/5% gates remain failed. A separately recorded learning/timing
continuation uses the user's standing acceptance of gradient differences around
8%, bounded at 10%, with forward differences bounded at 5%. This does not assert
quality equivalence. Fresh whole-model gates and longest-cohort memory preflight
still run before any update; their gradient ceiling is 10%, with strict results
retained. Stop if that envelope fails.

Columnwise V scales depend on future values within a 32-token block; the native
receipt separately reports this numerical effect. The perturbation test across
block boundaries remains exactly zero. Scales and calls stay sequence-local,
so this does not permit cross-example coupling. Do not claim exact BF16 causal
semantics or release this experimental backend without further quality evidence.

```bash
source .cache-runtime.env
.venv/bin/python -m experiments.b200_nvidia_mxfp8.training_screen \
  --config experiments/b200_nvidia_mxfp8/training02.yaml
```

Both conditions use the isolated cuDNN runtime, existing compiler/kernel caches,
identical data/targets/initial master and physical partitions. Candidate runs first;
reverse-order replication is unperformed. The FA4 control reuses its checksum-bound
unchanged kernel/packing validation, explicitly recording reuse. Outputs and logs
for the continuation are in their respective `training02/` subdirectories.
The first whole-model attempt stopped before updates at the one-token probe;
its failed logs and source snapshot remain in `training/`.

Completed `training02`: eager whole-model adapter-gradient relative L2 is
18.7114% against independent BF16 SDPA, exceeding both the strict 5% and
separate 10% learning ceilings. Losses 0.9492289424/0.9757985473 pass the loss
limit; all cross-example isolation effects are exactly zero. The run stops
before compiled checks, longest-batch preflight and optimizer updates. The FA4
control is unrun and no training-speed result is available. Keep BF16 FA4 as
the default; see [the finding](../../docs/findings/b200_nvidia_mxfp8.md).

The user subsequently requested a speed result regardless of that discrepancy.
`training03.yaml` authorizes a separate timing-only continuation, with 20 updates
per condition, ten warmup and ten measured. Both begin from the original master;
reuse all persistent caches and preserve prior failed receipts. Fresh eager and
compiled diagnostics retain strict/learning results and record explicit timing
acceptance separately. Numerical loss/gradient parity cannot stop this timing
screen, but sequence isolation, finite/missing gradients, memory preflight,
engine selection and matched input/partition identities remain mandatory. Stop
on failures of those execution checks. No promotion or quality validation follows.

```bash
source .cache-runtime.env
.venv/bin/python -m experiments.b200_nvidia_mxfp8.training_screen \
  --config experiments/b200_nvidia_mxfp8/training03.yaml
```

### Exact-shape cache preparation

The matched cohort records 267 distinct sequence lengths. NVIDIA specializes
both forward and backward plans on exact lengths/strides; serial first-use
preparation takes about 16 seconds per new pair in the live screen. A scoped
profile confirms CPU-side CuTe IR generation dominates fresh plan preparation,
and existing object-cache entries are successfully reloaded. This startup cost
must be reported separately from warmed update throughput.

Hypothesis: sixteen independent preparation workers reduce remaining cold-start
wall time by filling the same persistent cache before training encounters new
shapes. This intervention changes no attention inputs, kernels, configuration,
physical batches or model state. `prewarm.py` verifies the frozen manifest hash,
deduplicates its recorded lengths and builds exactly the selected D256,
16-query/4-KV-head forward/backward plans. It executes no attention or optimizer
updates. NVIDIA publishes object and record files atomically, allowing concurrent
cache readers/writers. The existing B200 has a 20.4-CPU quota and 234 GiB host
memory; use sixteen workers, leaving CPU headroom for training and retaining its
receipts. Stop a preparation worker on any compile/runtime failure; training
continues under its original finite-gradient checks. Start these workers during
warmup and ensure they finish before the measured updates.

```bash
source .cache-runtime.env
.venv/bin/python -m experiments.b200_nvidia_mxfp8.prewarm \
  --screen results/b200_nvidia_mxfp8/training03/summary.json --workers 16
```

Preparation receipts, source snapshot and worker logs go under that screen's
`prewarm/` directory. Compiler caches remain in the shared network-volume paths.

Completed timing-only candidate: 20 updates, last-ten mean 5.297811 seconds.
The user reminded us of the existing FA4 benchmark. Its initial master,
input hashes, objectives, optimizer/precision and all 20 physical batch contracts
match, so the redundant fresh FA4 repeat was intentionally stopped and the prior
4.086484-second control reused. MXFP8 is 29.64% slower in this comparison. The
candidate uses the isolated cuDNN runtime; no fresh same-runtime replication or
quality equivalence is claimed. The raw runner termination receipt remains
unchanged; `control_stop.json` explains the stop and
`reused_fa4_comparison.json` records the completed comparison separately.
Sixteen preparation workers finish in 208.068 seconds with 266 hits/268 misses
and no cache failures. Recorded lengths are preparation hints; materialized
training and diagnostic shapes may add plans. See
[the finding](../../docs/findings/b200_nvidia_mxfp8.md) for receipts and limits.
