# Direct FROST bindings for FP4 LoRA training

Hypothesis: bypassing cuDNN's per-call tensor/name/UID resolver reduces warmed
training host dispatch time. Keep the selected FP4 MLP forward/input-gradient
kernels, hardware packing, row scaling, BF16 rounding, live FP32 LoRA masters,
FlashQLA and FA4 unchanged. The intervention is opt-in and reversible; it wraps
only the six-operand, zero-workspace row-descaling plans in the native MLP cache.
The lowered NVIDIA executor retains its runtime guards. Both existing and newly
created plans use the wrapper, and original executors are restored on exit.

This first implementation reuses the original caller's packing, allocation and
view creation; it skips the vendor's operand-map resolution. It caches no
activation/output buffers or adapter values. Unsupported compiler identities,
operand contracts and workspace plans fail closed. Default recipes and pinned
source receipts remain unchanged until GPU evidence supports selection.

The user authorized retiring vLLM and running the native GPU screen. The pod now
uses the current refactored repository source; weights, datasets, environments
and shared caches are preserved. See the [native finding](../../docs/findings/b200_frost_training.md)
for passing CUDA evidence, timing qualifications and preserved failed attempts.

The probe rejects another live GPU process and never stops that process.
The `probe` checks the actual compiled FP4 LoRA MLP at
193/4096/16384 rows. Require bitwise output, input-gradient and all six adapter-
gradient agreement with original bindings, finite/missing-gradient checks,
independent outputs and changed-input/live-adapter replay. Check a second CUDA
stream and all four forward/dgrad K/N geometries. Compare ten alternating
synchronized complete forward/backward samples after six warmups; include all
packing, allocation and adapter work. Graph replay is a separate numerical gate,
not a timing claim about the bypassed Python work. Stop on any gate failure, OOM
or thirty minutes. No teacher calls, held-out selection or automatic promotion.

```bash
PYTHONPATH=src:. python -m experiments.b200_frost_training.probe --name native01
```

The full-model screen uses one resident worker, identical FP32 master,
RNG/data order, physical partitions and optimizer/scheduler resets. Reuse the
validated FP4 startup receipt and preserved strict failures. Require no new
plans/specializations/graphs during timed updates and >=2% lower warmed mean
complete-update time, with exact loss/gradient/update agreement. Reuse an
existing same-host training control; an old NC2 timing cannot attribute a gain
on the EU host. Use the frozen 320-row cohort and twenty complete
optimizer updates per trial; exclude updates 1--10 and retain all updates 11--20.
Run two FP4 controls to verify resident reset stability, then direct bindings,
then a final restored control. Targeted first-logical-batch validation requires
bitwise loss and all adapter gradients, unchanged masters and physical partitions.
Record host-call counts: existing CUDA graphs can bypass Python dispatch entirely.
Setup runs independent FLA, convolution and FlashQLA jobs concurrently, with
16 compiler workers for supported builds. Use the verified staged local runtime
when available; on-demand backend JIT calls still need separate precompilation
to overlap. Shared compiler/kernel caches retain their existing namespaces.
Keep the worker resident; reuse shared caches and overwrite only the systems
scratch adapter. Stop on nonfinite/missing gradients, workload drift, failed parity
or timed compilation. No ID evaluation or precision change is included.

```bash
PYTHONPATH=src:. python -m experiments.b200_frost_training.resident --session frostresident03 start --baseline-only
PYTHONPATH=src:. python -m experiments.b200_frost_training.resident --session frostresident03 submit --id 03direct --variant candidate
PYTHONPATH=src:. python -m experiments.b200_frost_training.resident --session frostresident03 submit --id 04control --variant baseline
```

The shared opt-in context is `training_frost_bindings()` from
`gleipnir.kernels.fp4.frost_bindings`. After the native and targeted model gates
pass, it can scope an existing Trainer's `train()` call. Its controller supports
`set_mode("original")` / `set_mode("direct")` between drained passes; original
mode restores both the class method and every wrapped plan executor. Exiting
the context restores the original path even after a failed native call.

`native03` passes all native gates. Median complete-MLP time falls 1.27/4.18/0.66%
at the three shapes; the 4096-token mean is dominated by a preserved control
outlier. The completed `frostresident03` whole-model screen finds no speedup:
pooled controls 3.39498 s/update versus direct 3.39585 s/update (0.026% slower).
First-batch gradients, loss histories and final adapter digests agree exactly;
all timed updates are warm. Keep the path opt-in and the recipe unchanged.
The worker remains resident; inspect its live receipt under
`results/b200_mlp_gemm/frostresident03/worker.json` before reuse. Timings,
startup failures and collection checks are in the linked finding.
