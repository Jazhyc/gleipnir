# FlashAttention 4 on the packed BF16 B200 recipe

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
the network volume and used by both conditions; these are warm-cache results.

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
