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
  only its identity and `serving_precision`. `run.py` merges on the scoring
  host with a fresh master-to-merged parity check, then scores through the
  shared optimized evaluator (optimized canary gate, FP8 dispatch audit).
- **27B.** `queue27b.sh` runs `ood27b.py`, the OOD version of
  `qwen38_27b_serving/id_drift.py`, with the recommended `graphs01` flags on
  the merge rebuilt by `remerge.py`.
- **Probe.** `mxfp8_probe.py` compiles an MXFP8 attention plan in
  isolation.

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

## Run on an EU-RO-1 B200 (2026-10-20)

Pod `lpjt8h7c2ccygh`: machine `2mj1yzztlz17`, `GPU-3d8d6bde…`, driver
595.91.07, AMD EPYC 9555, volume `cb7bxu71ug` mounted. The bound
`__init__.py` bytes were restored on the Pod, as for the B300.

### Gleipnir 4B v0.2: optimized minus BF16, −5.3 points on Mean-OOD pAUROC@20

| Gleipnir 4B v0.2, OOD | BF16 | Optimized | Change |
|---|---:|---:|---:|
| Mean-OOD pAUROC@20 | 0.8108 | 0.7579 | −0.0529 |
| Mean-OOD AUROC | 0.9199 | 0.9006 | −0.0193 |
| Pooled pAUROC@20 | 0.7514 | 0.6570 | −0.0944 |
| Mean Brier | 0.1339 | 0.1557 | +0.0218 |

By source (pAUROC@20): AgentDojo 0.9430 → 0.7913 (−0.152), SHADE-Arena
0.2653 → 0.1852 (−0.080), IaC 0.8845 → 0.8347 (−0.050), Rogue Eval
0.9804 → 0.9567 (−0.024), CUA-SHADE 0.7916 → 0.7805 (−0.011), MLE-Sabotage
1.0000 → 0.9990.

- **Gates.** Merged parity: mean difference 0.0009, correlation 0.99999.
  The optimized canary passed its gate (mean difference 0.0065, correlation
  0.9997; against the BF16 merge, 0.0073 and 0.9996).
- **Full-set drift.** Over the full OOD set, scores move by 0.057 on average
  (maximum 0.67), and 466 of 6,395 rows cross 0.5. The 20-row canary
  understates drift on long, low-margin OOD prompts.
- **Throughput.** 55.3M prompt tokens in 250 s, 222K prompt tokens/s.
- **Comparability.** The BF16 scores came from the B300 merge and the
  optimized scores from this host's merge. Both pass parity against the same
  FP32 master.

### Gleipnir 27B: optimized minus BF16, −0.25 points on Mean-OOD pAUROC@20

`opt27b_eu02` used the identical re-merge (rebuilt on the host that made the
campaign merge) and `QuantFullWorker` with the `graphs01` flags. The first
attempt (`opt27b_eu01`) hit the intermittent MXFP8 compile failure; the
retry ran through.

| Gleipnir 27B, OOD | BF16 | Optimized | Change |
|---|---:|---:|---:|
| Mean-OOD pAUROC@20 | 0.9087 | 0.9062 | −0.0025 |
| Mean-OOD AUROC | 0.9679 | 0.9723 | +0.0045 |
| Pooled pAUROC@20 | 0.9234 | 0.9219 | −0.0015 |
| Mean Brier | 0.0574 | 0.0564 | −0.0010 |

By source (pAUROC@20): AgentDojo 0.9983 → 0.9843 (−0.014), CUA-SHADE
0.8806 → 0.8741 (−0.006), IaC 0.9871 → 0.9851 (−0.002), SHADE-Arena
0.5875 → 0.5950 (+0.008); MLE-Sabotage and Rogue Eval unchanged. Mean
absolute score change 0.0255 (maximum 0.41), 108 threshold flips;
53.9K prompt tokens/s.

**Reading.** The same FP8/FP4/MXFP8 recipe costs the 27B a quarter of a
point and the 4B over five points. Most of the QAD headroom is in the 4B.
The 27B's optimized stack already almost matches BF16 without QAD. The 4B's
drift is concentrated in AgentDojo and SHADE-Arena. Its ID gap was not
measured here.

### Findings while getting there

- **Merges are not bit-portable across CPU vendors.** FP32 `b @ a` uses
  different BLAS accumulation on this AMD EPYC host than on the Intel hosts
  (B300, US-NE-1), so the BF16 shards differed. The receipts of the attempt
  that required identity are under `*_failed_merge_identity01`. The 27B merge
  was rebuilt on the host that made it originally and stayed identical.
- **The MXFP8 attention compile fails intermittently inside the server.**
  The error is the CUTLASS DSL `TYPE_UNSTABLE_JOIN` on the generated kernel
  `serving_exp2_d31b2d…`. It hit the 4B on US-NE-1 (driver 595) and the first
  27B attempt here, but not the 4B here or the second 27B attempt. In
  isolation the plan compiled for both geometries and six hash seeds. The
  cause is probably a race with concurrent DSL compilation in the server.
  Retries fix it; the robust fix is to compile the plan eagerly at worker
  load, which needs revalidation of the bound serving sources. A failed
  compile is a hard error, not a silent one.
- **Two stacks bind different versions of the same files.** The 4B binding
  check restores archived copies of `fp4_gemm_tune.py` and three canaries,
  while the 27B tuning receipt expects the repo versions. Restore them
  between runs.
- **Other recoverable failures** on the US-NE-1 B200 (Pod since terminated;
  receipts under `results/optimized_gap/us_ne1_b200/`):
  - inputs still being pushed;
  - an 8 October `server.json` from a wholesale-copied results directory;
  - the 27B base download first used a Python without `huggingface_hub`.
