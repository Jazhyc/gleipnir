# MXFP8 compiler compatibility repair

The frozen optimized run fails compiling a TMA producer loop: coordinates assigned
only inside conditional branches enter a dynamic loop with `None` on one path and
`Int32` on another. Test an opt-in derived-source repair that initializes the six
KV coordinates to `cutlass.Int32(0)` at producer entry. Every coordinate is assigned
by `_kv_tile_coords` before use; preserve all original statements, branches,
loads, masks, exp2 arithmetic and vendor sources. AST checks fail closed on source
drift. Derived-source hashing produces a separate compiled identity and retains
old caches and failed receipts. This is compiler compatibility, not promotion of
a new serving recipe.

Run `python -m experiments.filtered_fp4_baseline.compiler_repair --output PATH`
in the pinned serving runtime after retiring the BF16 scorer. Reuse the existing
native serving canary: causal/asymmetric/paged lengths, batch 128, 32K histories,
changed input/pages/lengths and CUDA-graph replay. Require finite outputs and
quantized-reference relative L2 <=0.01 in all seven checks; retain stricter BF16
precision failures separately. Record source/runtime/GPU bindings. Stop on
compile errors, source drift or arithmetic failures. No default runtime or frozen
campaign selection is changed by this opt-in entrypoint.

For an explicit optimized-serving candidate, use
`experiments.filtered_fp4_baseline.compiler_repair_worker.TypedCoordinateWorker`,
bind `serving_condition.mxfp8_coordinate_repair: typed_int32`, and set
`mxfp8_validation` to the new native receipt. Include this worker, repair entrypoint
and compatibility module in the candidate's source bindings. Preserve the selected
default and require fresh model score parity before full scoring.

BF16 ID scores provide the separate model-quality baseline. A successful native
test establishes compilation and operator arithmetic only; model score parity,
ranking and throughput require a subsequent explicit matched serving check.
