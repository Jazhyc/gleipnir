# Optimized-serving gap for the Gleipnir release candidates

Before QAD, measure how much OOD quality each release candidate loses when it
is served through its optimized stack instead of stock BF16. That gap is what
quantization-aware distillation has to recover.

- **Models.** Gleipnir 4B v0.2, the three-seed average
  ([seed average](../seed_average/README.md); BF16 OOD 0.8108), and Gleipnir
  27B (`qwen38-27b-s0-01`; BF16 OOD 0.9087).
- **Stacks.**
  - 4B: the selected optimized default (`serving_default.json`: FP8
    attention projections, FP4 MLP/GDN, MXFP8 attention).
  - 27B: `QuantFullWorker` with graphs to 2,048 tokens
    ([27B serving](../qwen38_27b_serving/README.md)).
- **Method.** No QAD, training or selection. The model is re-merged
  identically from the FP32 master, and the optimized canary gates parity.
  The frozen 6,395-row OOD suite is scored once and compared with the same
  model's BF16 scores.
- **4B context.** `prepare.py` copies the v0.2 BF16 OOD context and changes
  only its identity and `serving_precision`; `run.py` scores it.

## Attempt on a B300 (2026-10-20): blocked by the hardware gate

- The 4B identical re-merge succeeded on the EU-NL-1 B300.
- The scorer first stopped on runtime source binding. The optimized receipts
  pin `src/gleipnir/__init__.py` byte for byte, and the PATH helper added
  later changed it. For the evaluation, the bound bytes (`b6acfaad…`, the
  file before commit `5482f07`; the difference is PATH-only) were restored on
  the Pod; the repository keeps the fix. The two `_compare.py` diagnostics
  restore automatically from their archive.
- The optimized server then refused to start: `last-query kernel receipt
  hardware changed`. The stack's kernel receipts (last-query, FP4 tile
  tuning, MXFP8 and producer canaries, native audits) are bound to the B200,
  and the 27B `QuantFullWorker` reuses them.
- Receipts are preserved under `*_failed_source_binding01` and
  `gleipnir4b-v02-ood-optimized01`. No optimized OOD scores exist.
- **Next.** Run on a B200, the validated hardware and the deployment target
  for QAD. Restore the bound `__init__.py` bytes there too. The 27B also
  needs its base download, `qwen38_27b_serving/remerge.py`, and an OOD
  version of `id_drift.py`.
