# FlashAttention 4 on the packed BF16 B200 recipe

Default update, 2026-10-04: after reviewing the completed FA4 and FP4 screens,
the user explicitly selected BF16 LoRA with FA4 as the standard. The historical
screen contracts below retain their original SDPA control via
`qwen35_4b_b200_packed_sdpa`; this default change does not alter their timings.
See the [recipe decision](../../docs/decisions/b200_packed_bf16_training_recipe.md).
The earlier prospective replication criterion below is superseded by that user
selection; replication and held-out quality evidence remain uncollected.

The default integration smoke checks two ordinary updates from the frozen initial
adapter and cohort, reusing the completed checksum-bound FA4 receipt. Require
native FA4 metadata, no checkpointing, zero padding, finite gradients, a changed
FP32 master, and explicit `performed_this_run=false` reused checks. Stop on
receipt, kernel, input or master drift. It launches no teacher requests or quality
evaluation and makes no new warmed-throughput claim:

```bash
source .cache-runtime.env
.venv/bin/python -m experiments.b200_bf16_fa4.default_recipe_smoke
```

Artifacts: `results/b200_fa4_default_smoke/`; logs:
`logs/runpod/b200_fa4_default/`. Startup is monitored during this active turn;
there is no in-chat scheduler available for follow-ups after it ends.

Hypothesis: replacing the eight full-attention layers' segmented causal SDPA
calls with one native FA4 variable-length call per packed row reduces training
time now that the frozen base is unquantized BF16. The 24 FlashQLA layers,
FP32 rank-128 adapters, logical batch 32, 16,384-token packing budget, selected
decision-token projection, optimizer and no-checkpoint policy remain fixed.

Reuse the frozen 320-row seed-0 mixed cohort, targets and initial adapter from
the earlier systems screens. No teacher requests or held-out model selection
occur. This is a systems comparison; loss differences do not establish quality
equivalence. The current default stays SDPA until this screen completes.

Run SDPA and FA4 from the same initial adapter for 20 ordinary Trainer updates.
Report startup, the complete loop, every synchronized update, and the final ten
updates after a complete ten-batch warmup. Compare complete matched physical
partitions and token counts; never discard individual slow measured batches.
Require at least 5% less time in both the measured ten-update window and complete
Trainer loop before recommending a follow-up. Reverse-order replication is
required before adopting a promising candidate. Shared caches are retained on
the network volume and used by both conditions. Complete any cache restoration
before either condition starts; a run overlapping cache writes cannot establish
a warm-cache comparison.

The SDPA baseline reuses the completed default smoke validation, recording its
checksum and skipped checks. FA4 changes the kernel family and therefore requires
fresh native BF16 GQA forward/backward checks, eager/compiled model packing
isolation and parity checks, longest actual cohort batch memory preflight, and
finite/missing-gradient checks at every update. Stop on a failed gate, input or
initial-master drift, version mismatch, fallback, compile-limit exhaustion,
nonfinite values or OOM. Keep failed receipts; do not relax tolerances. Native
FA4 uses the existing isolated 4.0.0b33 overlay and CUDA-13 CuTe 4.8.0 wheels.
The combined process retains FlashQLA's pinned TVM FFI 0.1.11 ahead of FA4's
overlay. Its native canary verifies both imports and kernels before model loading.
The initial FA4-first overlay attempt and its native receipt are preserved under
`results/b200_bf16_fa4_overlay_conflict/`; no training comparison follows from it.

```bash
bash experiments/b200_bf16_fa4/launch.sh
```

Artifacts: `results/b200_bf16_fa4/`; logs: `logs/runpod/b200_bf16_fa4/`.
The scoped packing router is opaque to compilation for both backends. Its
metadata explicitly records the effective full-attention kernel even though
the model's Transformers routing key remains `sdpa`. Unpacked diagnostic calls
retain the SDPA reference. No full training campaign or release is launched.

## Completed screen, 2026-10-04

The combined-process native FA4 forward/backward canary passed, but ordinary
Trainer eager packing parity failed: adapter gradient relative L2 0.0828666
against the unchanged 0.05 gate. FA4 performed zero optimizer updates. Keep the
SDPA default; there is no valid FA4 throughput or quality conclusion.

The SDPA control completed 20 updates, but restoration of older compatible
compiler-cache entries overlapped its measured window. Preserve all timings as
exploratory. See the [finding](../../docs/findings/b200_bf16_fa4.md) for runtime,
input identity, failed attempts and cache limitations.

The failed-receipt writer is repaired. A separate zero-update replay retains
the complete eager receipt without replacing the original run:

```bash
source .cache-runtime.env
.venv/bin/python -m experiments.b200_bf16_fa4.replay_failed_gate
```

Replay artifacts/logs use `b200_bf16_fa4_eager_gate_replay`. The replay verifies
input and initial-master identity and sets `canary_only=true`. Its output
directory must be absent; completed receipts cannot be overwritten accidentally.

After reviewing the full receipt, the user explicitly authorized continuing with
the approximately 8% difference. `accepted_config.yaml` records this instruction
and a 10% learning-acceptance ceiling. Strict 5% results remain visible; leakage,
loss and finite-gradient requirements stay fixed. Both timing conditions restart
from the same initial adapter after the cache merge has completed:

```bash
source .cache-runtime.env
.venv/bin/python -m experiments.b200_bf16_fa4.run \
  --config experiments/b200_bf16_fa4/accepted_config.yaml
```

The continuation writes `b200_bf16_fa4_accepted` artifacts/logs and reuses the
unchanged native kernel receipt. Fresh compiled packing and memory checks still
precede FA4 updates. Acceptance does not select FA4 as the default.

The continuation completed both 20-update trajectories. Final-ten mean time is
5.12363 s (SDPA) vs 4.08648 s (FA4), a 20.24% reduction with identical physical
partitions/tokens. Peak allocation is 146.995 vs 145.242 GiB. Compiled gradient
relative L2 is 0.0517730, accepted separately from strict failure; all isolation
checks and longest-batch memory preflight pass. Fresh FA4 checks front-load
compilation: complete invocation is 567.05 vs 507.40 s. Its 72% shorter Trainer
loop is not an equivalent startup improvement. Reverse-order replication and
quality validation remain unperformed; the default stays SDPA.
