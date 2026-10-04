# Monitor-directed injection augmentation at 9B

Date: 2026-10-04. Status: training and evaluation complete; artifacts collected
and independently verified. Authorized B200 shutdown is the remaining closure step.

The same intervention improves clean ID ranking at 9B, but transfers much less
strongly to APPS injections than at 4B. The fixed final checkpoint gives:

| Metric | Original regular 9B adapter (cached) | Augmented 9B adapter |
| --- | ---: | ---: |
| Clean ID source-macro pAUROC@20 | 0.879559 | 0.893869 |
| Clean ID pooled AUROC | 0.952174 | 0.958272 |
| Clean ID pooled Brier | 0.085105 | 0.081521 |
| Clean ID pooled log loss | 0.283268 | 0.276137 |
| Clean ID pooled ECE | 0.036282 | 0.044700 |
| APPS injected-honest FPR, template mean | 99.85% | 83.36% |
| APPS injected-backdoor TPR, template mean | 100% | 98.25% |
| Matched injected-backdoor vs injected-honest AUROC, template mean | 0.658763 | 0.678591 |
| Vanilla APPS AUROC | 0.931378 | 0.912592 |
| Vanilla APPS TPR at clean-code ~2% FPR | 63.21% | 55.49% |

Each adapter uses its own threshold calibrated on the same 655 clean honest
controls: 0.132964 original and 0.148047 augmented, both giving 1.985% FPR
with strict score > threshold. Keep that threshold fixed for injected examples.
Template averages reuse the same problems; they are not independent samples.

| Injection | Augmented injected-honest FPR | Augmented injected-backdoor TPR | Augmented matched injected AUROC |
| --- | ---: | ---: | ---: |
| transfer_gpt4 | 100% | 100% | 0.622239 |
| transfer_gpt41 | 100% | 100% | 0.538408 |
| transfer_gpt4o | 50.08% | 94.74% | 0.875125 |

Two templates still produce universal injected-honest alarms. GPT4.1 matched
injected AUROC also falls from 0.625040 to 0.538408. The improvement is therefore
concentrated in GPT4o, rather than broad immunity. The 4B counterpart reached
44.43% mean injected-honest FPR and 0.859471 matched injected AUROC; this 9B
replication does not reproduce that degree of OOD robustness. Vanilla APPS
ranking and detection also regress, despite the clean ID gain. No promotion follows.

Clean ID source-macro ranking improves within both sources: Gloom pAUROC@20
0.811737 to 0.837760 and STRIDE 0.947380 to 0.949977. A common monotonic score
shift cannot explain changed AUROC. Reduced reliance on nuisance text and
augmentation regularization are plausible mechanisms, but this one-seed
comparison does not identify previously unknown shortcuts or isolate the cause.
Calibration improves by Brier/log loss and worsens by ECE. Unique pooled scores
fall from 1,645 to 1,452; ranking metrics account for ties.

For the new adapter's paired synthetic ID views, pooled AUROC is 0.958272 clean
and 0.958053 injected. Project-normalized source-macro pAUROC@20 is 0.893869
and 0.892182. Accuracy at 0.5 remains 89.38%. Harmless FPR increases from 1.92%
clean to 2.56% injected at source-specific clean ~2% FPR thresholds; harmful
TPR is 81.22% clean and 81.97% injected. Mean absolute score changes are 0.0163
for harmless and 0.0177 for harmful examples. Correct-to-wrong flips at 0.5 are
0.71% harmless and 0.44% harmful; wrong-to-correct flips are 0.14% and 0.93%.

Injected-harmless FPR is 2.03% for user additions, 2.24% assistant and 3.40% tool;
2.85% for channel spoofing and 2.28% for decision pressure. Agreeing/conflicting
demand groups give 1.70%/3.44% harmless FPR and 84.53%/79.38% harmful TPR.
Demanded-1 additions shift mean scores upward more than demanded-0 additions,
which suggests residual demand sensitivity; groups contain different trajectories,
so this is not a counterfactual causal comparison. Effects remain small compared
with APPS score shifts. There is no original-adapter synthetic baseline.
Historical synthetic `pauc20` fields are sklearn-standardized; use the separately
audited project partial area divided by 0.2 for clean/injected comparisons.

All 15,138 fresh raw scores pass exact input coverage, identity, label/source,
rendered prompt hash, one-token decision and raw-logprob reconstruction checks.
Maximum prompt lengths are 16,321 benchmark, 3,931 honest controls, 29,513 clean
ID and 29,546 synthetic ID, within the 32,768-token serving envelope. All 256
master tensors are finite FP32 and exactly equal to their rebased serving tensors.
The 20-example serving parity gate passed: augmented correlation 0.999793,
mean absolute difference 0.003068; base correlation 0.999665, difference 0.005366.
Maximum serving/reference adapter effects are 0.785448/0.795698.

The final remote inventory contains 58 files and 2,864,290,636 bytes, including
master/serving weights, checkpoint, raw scores and logs. Every collected file
matches its checksum before local report recomputation. Exact remote reports
are retained in `evaluation_regular/remote_snapshots/`; recomputation differs
only in the absolute benchmark filename. Final receipts are
`evaluation_regular/collection_verification.json` and
`artifact_inventory_verification.json`. The serving engine and launcher exited,
and the remote inventory and preservation checks confirm no GPU process remains.

The authorized replication freezes the completed 4B replacement intervention
and uses the validated checkpointed Qwen3.5-9B recipe at 5e-5 for one epoch,
272 updates. The unchanged 8,688 views contain 3,475 injections and 5,213 clean
examples. Evaluation uses the same regular instruction and 15,138 fresh
examples: APPS 5,184, honest controls 3,930, clean ID 3,012 and injected ID 3,012.
See the [decision](../decisions/monitor_injection_augmentation_9b.md) and
[protocol](../../experiments/monitor_injection_augmentation_9b/README.md).

The cached original 9B regular baseline was audited against all canonical
input prompts, IDs, labels and decision logprobs. ID source-macro pAUROC@20 is
0.879559. APPS injected-honest alarms are 100%, 100% and 99.54% for GPT4,
GPT4.1 and GPT4o transfer conditions at its clean-code ~2% FPR threshold.
The original adapter will not be rescored. The frozen baseline JSON SHA256 is
`1cb70008f44576f684fa5284eb9bd2612e9655dcf4a8b011064a8fa83afd5fa0`.

The same-model startup reference is the completed regular 9B metadata, SHA256
`ad97d1e02160084c370445625e5db1d13ee32a88fbe5be6720df030c24fc8b98`.
Fresh seed-0 initialization must match its initial tensor digest
`5ba0b3d6e44c20a1a1ccfd2d5923b6d90a5dd930e02321ff832cb471984bcf79`.
All-layer nonreentrant checkpointing is reused alongside packed BF16
FlashQLA/SDPA and the persistent compiler cache. Strict historical failed
gradient comparisons remain negative; startup probes are marked reused.

Six focused checks passed locally and remotely: cached prompt/label/logit
integrity, frozen recipe and command wiring, acceptance of the recorded
checkpointed reference, rejection of mismatched live/reentrant checkpointing,
and historical failure preservation. Ruff, shell syntax and diff checks passed.
Preparation preserved a schema-check failure receipt: original APPS
`prompt_sha256` hashes the source prompt, while ID `margin_prompt_sha256` hashes
its rendered boundary. Both historical contracts now validate without changing
any scores or input bytes. All 28 frozen source files match on the B200.

User-authorized shutdown follows complete evaluation, artifact collection and
verification. There is no separate in-chat heartbeat scheduler;
active goal execution can inspect progress but a remote queue is not an agent
follow-up. The goal remains active through verified shutdown.

The source-checked launch started PID 15742 on B200 `3422tdq7tfog52`; training
worker PID 15745 completed and exited. Model loading verified 8,953,803,264 frozen BF16
elements and 232,783,872 trainable FP32 elements in 256 tensors. Live all-layer
nonreentrant checkpointing matches the recorded recipe. All 24 GDN layers bound
to FlashQLA with startup diagnostics explicitly marked reused/not repeated.
The populated shared compiler cache resolves to
`/workspace/gleipnir/.cache/training/student_injection_awareness`.
Preparation manifest SHA256 is
`8ea9596a8458520525747c31b19eb248549a510b087e5ffd45c94baf4fce4442`.
Launch, cache, training and evaluation receipts are collected locally. An initial
launcher import-path failure started no workload and is preserved separately; the source-checked launch
uses an explicit repository import path.

At startup, ordinary training reached 2/272 completed updates, with mandatory
finite/missing-gradient guards permitting both updates. First-update elapsed
time was 282.16 seconds including compilation; the next update took about
20 seconds. GPU utilization reached 100%, with 37,470 MiB used. These early
timings do not establish whole-epoch throughput. The numerical training command
matches the original 9B regular condition; only campaign paths and the explicit
startup-validation reference differ. Feature implementation is committed as
`ac21e87`; the full goal remains active until verified shutdown.

Training subsequently completed all 272 updates and one epoch at 5e-5, with
mean loss 0.226887 and training runtime 6,264.15 seconds (104.4 minutes).
Warmed updates averaged approximately 22 seconds. Peak CUDA allocated/reserved
memory was 35.74/36.55 GiB. These are execution diagnostics, not quality results.
The collected metadata passes the matched validator, including the expected
initial tensor digest, original teacher targets, checkpointing, objective,
precision and startup-reuse contract. The FP32 master, rebased serving artifact
and final checkpoint are collected locally. Master/serving file SHA256 values
match the completed training receipt:
`36059afaff50c119c86f039860923a08d9672e9807aca261b784b88bf9afb283` and
`64ff99a3abf331875d1c973f833dadb89c5462b918d94c17413f7ce3c2236e33`.
Bounded master-reference scoring completed; persistent-vLLM startup and its
adapter-specific parity gates precede population evaluation.

Benchmark metrics are identical between remote revision
`29b31d9bea6b0072e3710f47d09299281c7639d0` and local revision
`b55c2014d4c22ceea810b2639e6c0f672d83eccb`, both on
`feat/gleipnir-vllm-grid`. The revision difference affects only three documentation
files. Four metric/imported-helper checksums match; their provenance receipt is
`results/monitor_injection_augmentation_9b/benchmark_metric_provenance.json`.
