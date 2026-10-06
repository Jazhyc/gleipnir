# B200 attention and GDN throughput campaign

Started 2026-10-06 03:19 UTC. The user authorizes approximately nine hours of
attention/GDN optimization, prioritizing throughput at high concurrency, FP8
first, alternative kernels and small integration changes. Stop the old server
before changing kernels; keep one serving engine on the GPU. Capacity is already
authorized. If sensible options are exhausted, the user authorizes terminating
the B200 after verified artifact collection. No public deployment is requested.

Hypothesis: native low-precision attention and projection kernels improve
complete monitor input-token throughput after FROST FP4 MLP optimization.
The selected control is `results/b200_frost_inference/frost02`; preserve its
timings/predictions and validated source/runtime/merge receipts. Use the same
FP32-master-derived ephemeral merged BF16 model, frozen prompts and one-token
binary scoring. Retain BF16 non-target operations, no prefix caching, and
network-volume compiler caches.

First establish the FROST control at a production-sized envelope: 90% GPU memory,
128 engine sequences, 32,768 scheduled tokens and 32,768 context. This envelope
differs from the recorded 25%/16-sequence control and needs one matched baseline;
reuse it for all compatible candidates. Include the original quick 64-row,
two-repeat c1/4/16 screen and a full 320-row two-repeat c16/32/64/128 load sweep.
The latter is the original systems cohort (1,310,581 prompt tokens), with frozen
seed-0 order and labels, sufficient requests to sustain the higher concurrency.
Client concurrency and engine sequence capacity are separate; report both.
Record the highest measured sustainable throughput, all repeat durations and
latency percentiles. A concurrency increase alone is not a kernel speed gain.

Candidate order, revised when evidence warrants:

1. FlashInfer/TRTLLM FP8 Q/K/V full attention and FP8 KV storage, with FROST MLPs
   and BF16 GDN fixed. Verify actual FP8 query/cache dispatch and causal D256 GQA.
   Freeze and record appropriate scales; count packing/conversion costs.
2. Alternative native attention kernels, including NVFP4 KV with FP8 query and
   compatible BF16 FA4, if the first attention kernel does not help. Preserve
   causal ragged semantics and paged cache/graph compatibility.
3. Native FP8 GDN QKV/output projection GEMMs, leaving small gate projections,
   convolution and recurrence operands BF16 and gates/state FP32. Test FP4
   projections separately if warranted by native kernel availability.
4. Alternative GDN prefill backends (FlashQLA and CuTe DSL) and targeted fusion,
   dtype/conversion or dispatch changes guided by profiling. An FP8 recurrence
   needs kernel support; do not advertise FP8 state storage as FP8 computation.
5. Combine only measured improvements, then examine remaining bottlenecks and
   capacity/token-budget tuning at the strongest high-concurrency condition.

Before a new precision/backend trial, inspect startup closely, audit loaded
scope/runtime and actual attention/GDN dispatch, retain finite/missing-output
checks and perform the unchanged adapter-specific twenty-row score canary.
Reuse unchanged FROST arithmetic validation when possible; check changes in
the supported batching envelope. Keep a finite failed score canary as diagnostic
and report the failure without widening limits. Stop on nonfinite/missing
outputs, native arithmetic/scope failure, source/input drift, truncation, OOM
or unsupported graph launches. Diagnose and preserve failed attempts before
retrying; an observation timeout alone never justifies restarting a live run.

Every condition reports pooled/per-source AUROC, equal-source macro over
dual-label sources, undefined groups, repeat variation, score/margin drift,
threshold flips, ties, calibration and partial AUROC. Bind labels to exact
prompt identities. These training-seen development sets do not establish
held-out quality; no final ID promotion or silent acceptance of degraded
serving precision. A >5% matched high-concurrency gain merits confirmation;
retain smaller and negative results. Warmed confirmation reuses the same
worker and passed unchanged canary rather than reloading or rerunning a control.

Write a condition configuration, intervention and matched reference before each
launch. Archive executed code/configs, receipts, predictions, metrics and logs;
commit completed features and findings after focused tests and successful
startup. Collect important artifacts after each phase and before termination.
Keep source weights/master adapters/caches on the network volume; merged models
and reconstructable packed weights can remain ephemeral.

Check startup every 30–60 seconds during active goal work, then progress and GPU
health at least every ten minutes. This session has no timed agent heartbeat
scheduler; active goal turns support checks but cannot promise timed wakes after
a turn ends. Record live PIDs, configuration and artifact paths for continuation.

Outputs: `results/b200_attention_gdn_serving/`; logs:
`logs/runpod/b200_attention_gdn_serving/`. All conditions use the existing NC2
B200, no concurrent control server or newly launched capacity.
