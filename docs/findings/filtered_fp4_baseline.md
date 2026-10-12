# Filtered FP4/FA4 baseline campaign

The user authorized setup and a new optimized-training baseline on their NC2
B200 on 2026-10-12. Pod `fs334v0u4zysvn` mounts retained network volume
`ixbh81vf9c`; direct SSH was enabled through the account proxy without a restart.
GPU UUID is `GPU-480a554d-fe81-e7b9-bd55-02ca0244f4aa`, with 183,359 MiB VRAM
and driver 580.126.09. Connection metadata is in `.runpod/qat-nc2.json`.

The pinned training/serving runtime archive passed SHA256 verification and was
restored to its original paths. Dependency overlay links, native serving sources
and container caches were restored from the retained archives. The FP4 runtime
verifier reports Torch 2.11.0+cu130, Transformers 5.14.1, FA4 4.0.0b33,
Triton 3.7.1, PEFT 0.19.1 and cuDNN 9.26/frontend 1.31.0 with the expected
projection-GEMM source hash. Existing shared compiler caches are reused.

[`filtered_fp4_baseline`](../../experiments/filtered_fp4_baseline/README.md)
defines the frozen contract. Preparation binds 86 prerequisite artifacts and
removes exactly 852/8,688 rows, retaining 7,836 rows, 75,851,675 untruncated
tokens and 245 updates. Original Kimi soft targets and zero-B initialization
are unchanged. The existing filtered SDPA ID predictions are the comparison;
no redundant control training, OOD evaluation or QAT algorithm is introduced.

The detached shared campaign driver launched as PID 574 (Trainer PID 651), chaining
training, master reference, ephemeral BF16 merge, merged reference and optimized
serving/ID evaluation with required new-adapter parity. Launch and frozen input
receipts are collected locally under `results/filtered-fp4-fa4-baseline01/` and
`data/filtered-fp4-fa4-baseline01/`; stage logs remain on the volume under
`logs/runpod/filtered-fp4-fa4-baseline01/`. This launch receipt is not a completed
quality result. No in-chat scheduler is available; the Pod stays running.

Training completes all 245 updates and the exact one-epoch population. Trainer
runtime is 2,336.893 seconds (38 minutes 57 seconds), mean training loss is
0.1944368, and steady update timing averages 9.0347 seconds after the first
86.6203-second update. Initial GPU memory is about 146 GiB. The completed FP32
master SHA256 is `4eb97e5385f53b135c0b77cb92d0cbbfa78a661100f9ba8ac4205dbabb34f057`.
Master-reference scoring, merge and merged-reference finish. Master/BF16-merge
parity passes: score MAE 0.0011047, maximum difference 0.0074601 and correlation
0.99998785, with nonzero adapter effect 0.87237.

Optimized evaluation fails before producing canary predictions or ID metrics.
The MXFP8 attention kernel's first compilation raises CUTLASS
`DSLUserCodeError[TYPE_UNSTABLE_JOIN]`: `k_row` has type `None` on one path and
`Int32` on another in generated `serving_exp2_d31b2dfc...py:1319`.
The engine exits, producing HTTP 500 at the score endpoint; the client connection
errors are secondary. The loaded CUTLASS distribution reports pinned 4.8.0.
This is an execution failure, not evidence of failed optimized score parity.
The campaign preserves its failed evaluate stage and retires the scorer; no GPU
compute process remains at the check. Training is not repeated and no arithmetic
fallback or parity waiver is applied. ID baseline completion requires fixing and
validating this serving compilation failure with the same preserved adapter.
Completion metadata, passing merge parity and failure logs are collected locally.

## BF16 serving follow-up

At the user's request, `filtered-fp4-fa4-bf16-serving01` reuses the verified
adapter, master reference, merge and merged-reference receipts, with no training
or merge repeated. All 3,012 frozen ID examples complete under regular BF16
vLLM serving. Fresh master/serving MAE is 0.0010124 and correlation 0.99998745;
the BF16 projection/attention audit passes. Scoring takes 299.362 seconds for
33,750,959 prompt tokens, or 112,743 prompt tokens/second. This is one ordered
c128 pass, not a repeated throughput benchmark.

| ID metric | FP4/FA4 training, BF16 serving | Filtered SDPA control, BF16 serving |
| --- | ---: | ---: |
| Source-macro raw normalized pAUROC@20 | 0.887045 | 0.895914 |
| Source-macro AUROC | 0.962825 | 0.963772 |
| Pooled AUROC | 0.956521 | 0.953400 |
| Pooled raw normalized pAUROC@20 | 0.855490 | 0.855463 |
| Pooled Brier | 0.082633 | 0.108310 |
| Pooled ECE | 0.047501 | 0.095864 |
| Pooled recall at 0.5 | 0.904851 | 0.943408 |
| Pooled FPR at 0.5 | 0.143875 | 0.297009 |

Gloom pAUROC is 0.818870 versus 0.828524; STRIDE is 0.955219 versus 0.963304.
The combined training recipe loses 0.887 percentage points of source-macro
pAUROC, while calibration and fixed-threshold false alarms improve. One seed
and historical source/host qualifications remain; attention and MLP precision
change together. The copied comparator description in the frozen BF16 YAML
retains the original optimized-serving wording; actual serving in this follow-up
and the control is BF16, as audited above. No ID selection or promotion occurs.

Setup-only quoting, adapter-link and missing-completion-receipt failures are
retained. The evaluation stage had not scored any rows when the missing receipt
failed; its orphan server was verified and retired before recovery. The corrected
evaluation completes, with `recovery_status.json` binding all outputs while the
original failed status/log remains. Predictions, calibration, threshold/tie
diagnostics, source bindings, audits and reuse receipts are collected locally.
The BF16 scorer is retired before the authorized compiler checks; the Pod stays
running.

## Compiler compatibility repair

The opt-in [repair](../../experiments/filtered_fp4_baseline/compiler_repair.md)
initializes six TMA KV coordinates to `cutlass.Int32(0)` at producer entry.
All coordinate values are overwritten before use. AST checks preserve every
original statement and reject source drift. Vendor sources, the frozen default,
historical receipts and old compiled caches remain unchanged; derived-source
hashing creates a separate kernel identity.

The B200 native test compiles and all seven finite/quantized-reference checks
pass, including causal/paged/asymmetric lengths, batch 128, 32K histories,
changed pages/lengths and graph replay. Maximum quantized-reference relative L2
is 0.0016667 (limit 0.01). The stricter BF16-relative 0.05 precision gate still
fails, with a maximum 0.0533631; retain `passed=false`, `arithmetic_passed=true`
and `diagnostic_only=true` separately. The receipt and logs are under
`results/filtered_fp4_compiler_repair/native01.json` and matching Runpod logs.
`native02.json` refreshes the final source binding after removing a trailing
blank line, reusing the identical derived kernel; AST identity and the original
model-test source are archived beside the receipts.
A bounded actual-model serving check is separate from this operator validation.

The repaired optimized server completes three c4 passes on the same 20
training-source canary rows (51,132 prompt tokens). The final pass passes
master/serving parity (MAE 0.0058295, correlation 0.99980557) and BF16-serving
agreement (MAE 0.0053273, correlation 0.99982085). All three passes retain pooled,
macro and each of five sources' AUROC 1.0, a zero AUROC deviation from the saved
BF16 canary. This easy training-seen cohort does not establish ID/OOD equivalence.
Canary Brier varies 0.0067544–0.0116641 across passes; only the final raw prediction
set is archived, so do not claim all-repeat score parity. First-pass throughput
is 16,006 prompt tokens/s; the two warm passes are 51,462 and 48,483 prompt
tokens/s. These short c4 diagnostics are not comparable to the c128 full-ID
throughput above. The candidate retires after the check, with no GPU compute
process remaining. The original serving selection is preserved; the tested repair
is available through the explicitly bound `TypedCoordinateWorker` candidate.

## Full repaired optimized-serving ID comparison

At the user's request, the [optimized ID contract](../../experiments/filtered_fp4_baseline/optimized_id.md)
freezes two ordered passes over the same 3,012 examples, reusing the completed
adapter/merge and `native02` repair. All three new startup canaries pass the
master gate (MAEs 0.0058295, 0.0058295, 0.0119203; correlations 0.99980557,
0.99980557, 0.99750763). All raw canary repeats are retained. Both full ID passes
complete without compiler or scoring failures; native audits confirm MXFP8
attention and FP8 attention projections.

| ID metric | Same-adapter BF16 | Optimized pass 1 | Optimized pass 2 |
| --- | ---: | ---: | ---: |
| Source-macro raw normalized pAUROC@20 | 0.887045 | 0.850308 | 0.847754 |
| Source-macro AUROC | 0.962825 | 0.948017 | 0.947345 |
| Pooled AUROC | 0.956521 | 0.941193 | 0.940109 |
| Pooled raw normalized pAUROC@20 | 0.855490 | 0.812779 | 0.808482 |
| Pooled Brier | 0.082633 | 0.097521 | 0.098278 |
| Pooled ECE | 0.047501 | 0.054901 | 0.058461 |
| Gloom raw normalized pAUROC@20 | 0.818870 | 0.742780 | 0.738401 |
| STRIDE raw normalized pAUROC@20 | 0.955219 | 0.957835 | 0.957107 |
| Prompt tokens/second, c128 | 112,743 | 179,427 | 177,828 |
| Scoring seconds | 299.362 | 188.104 | 189.796 |

The mean macro pAUROC loss is 3.80 percentage points, concentrated in Gloom;
STRIDE is slightly better. The optimized recipe is about 1.58 times faster.
The 0.255-percentage-point macro pAUROC repeat range is smaller than the
BF16-to-optimized gap. Repeated optimized scores have MAE 0.0137278,
correlation 0.9960503 and maximum difference 0.3535179; 41/3,012 decisions
flip at 0.5. BF16 repeat variation is unmeasured. This compares the combined
serving precision/kernel recipe on fixed weights, not isolated projection
quantization or training effects. No held-out selection or promotion occurs.

Raw predictions, batch timing, source/command bindings, dispatch audits,
calibration/threshold/tie diagnostics, parity gates and repeat comparison are
collected under `results/filtered-fp4-fa4-optimized-id01/`. The candidate is
retired and the GPU has no compute process; the Pod remains running. The fixed
BF16 baseline and failed original optimized receipt are preserved.

Verification: 31 campaign/filter/source checks pass; log is
`logs/local/filtered_fp4_baseline/bf16_compiler_tests.log`. Ruff passes. No training
implementation changed. Native experimental-API and shutdown semaphore warnings
are retained in the logs; both tested model-server processes are verified retired.
