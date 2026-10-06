# FP4 GDN projections as the B200 serving baseline

Decision date: 2026-10-06. The user explicitly accepts the source-macro AUROC
change in percentage points and asks to add the FP4 GDN projection variant to
the stack, measuring further improvements against it.

Use FROST native FP4 for the 64 MLP projections and the 48 large GDN
`in_proj_qkvz`/`out_proj` GEMMs. Keep small GDN gate projections BF16,
convolution and recurrence operands BF16, gates/state FP32 and full attention
BF16 FlashInfer/TRTLLM. This is projection quantization; the delta-rule
recurrence does not perform FP4 or FP8 matrix arithmetic.

Preserve the FP32 master adapter and its ephemeral merged BF16 checkpoint.
Pack frozen forward weights at startup; reuse shared compiler caches and the
loaded worker for compatible comparisons. The matched envelope is one B200,
90% memory, 128 engine sequences, 32,768 scheduled tokens/context, one decision
token and disabled prefix caching. Stop the previous server before active
kernel changes. This decision changes inference optimization, not training or
any frozen past evaluation contract.

## Evidence and acceptance

`results/b200_attention_gdn_serving/fp4_gdn_projection02` completes six quick
and eight full-workload passes. All 64 MLPs retain the original native checks;
all 48 GDN projections have the intended FP4 scope. Six independent decoded
GDN references at M=1/17/129 have maximum relative-L2 error 0.00000753.

Full 320-row throughput medians at c16/32/64/128 are approximately
133,779/167,276/170,163/159,619 input tokens/s. Relative to the matched
FP4-MLP/BF16-GDN control, the gains are 2.48%/12.18%/15.20%/11.89%.
Five additional c128 passes reuse the same worker and unchanged checks,
confirming a median 165,927 input tokens/s, +16.31% against the archived control.
A client transport failure is retained; three complete passes were preserved
and two missing passes resumed without reloading the GPU worker.

At c128, full-cohort source-macro AUROC changes by −0.004659 (−0.47 percentage
points), while pooled AUROC changes by +0.014690 (+1.47 points). Mean/max score
drift is about 0.053/0.510, with sixteen threshold flips. These are training-seen
systems-development examples, not held-out quality evidence.

The twenty-row strict score canary fails (mean error 0.024425, correlation
0.992690 against FROST). Preserve `http_parity.json` and the original diagnostic
classification. The separate `quality_acceptance.json` records the user's
explicit `user_accepted_finite` selection; do not relabel the strict failure as
a pass or widen its limits. Missing/nonfinite outputs remain disallowed. This
acceptance applies to the current adapter and pinned recipe; future adapters
and arithmetic changes retain their own serving checks and AUROC reporting.

## Reuse

`experiments/b200_inference_benchmark/baseline.json` binds the result summary,
workload, executed recipe, full sweep, confirmation and quality-acceptance
checksums. The prior selection remains under `baselines/`; explicit historical
references are unchanged. `experiments/b200_attention_gdn_serving.run` defaults
to `fp4_gdn_projection.json`; new comparisons resolve the selected baseline.
Reusing this selection requires no control replay or server restart.

```bash
python -m experiments.b200_attention_gdn_serving.run --output fp4_baseline_run
```

The earlier CuTe GDN recurrence alternative passes isolated checks and the
small model canary but produces NaN scores on the full workload. It remains
rejected and is not part of this baseline.

Further kernels retain strict master-parity diagnostics, but a checksum-bound
accepted baseline may be used as the numerical change reference with the same
mean-error/correlation limits. This does not waive adapter-effect or finite-output
checks, and does not accept additional kernel drift automatically.
