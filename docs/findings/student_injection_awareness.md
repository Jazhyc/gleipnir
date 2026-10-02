# Student prompt-injection awareness campaign

The [prespecified experiment](../../experiments/student_injection_awareness/README.md)
compares regular and explicitly injection-aware student instructions at 4B and
9B, holding all 8,688 tool trajectories and cached Kimi K3 targets fixed. It
changes the instruction during both training and evaluation, so a difference
cannot be attributed solely to learning or solely to test-time elicitation.
Final one-epoch checkpoints and canonical ID/OOD populations are fixed before
evaluation. No task-quality result is available yet.

## First 4B startup, 2026-10-02

The regular condition uses pinned Qwen3.5-4B, BF16 packing without activation
checkpointing, FP32 rank-128 adapters, seed 0 and learning rate 5e-5. The initial
adapter tensor digest is
`a6b1d2e9fd89efff9523150a76035a2e5d27900eaae3c7a4820e3b9277078f11`.
All 128 initial LoRA B tensors (99,090,432 elements) are zero, so the initial
adapter contributes no learned update to the base model.
All 8,688 inputs were audited: 83,816,369 tokens, maximum 29,337, zero truncation
against the 29,696-token training cap.

Strict eager and compiled packing canaries passed, including trajectory
isolation. Adapter-gradient relative L2 differences were 0.00745419 and
0.00820815, respectively, against the existing 0.05 limit. The longest-actual-
input preflight backpropagated 32 examples totaling 922,511 tokens, with finite,
nonzero gradients and unchanged master weights. Peak allocated GPU memory was
160,056,479,232 bytes (149.06 GiB) on the existing single B200.

Two separate strict comparisons failed and remain recorded as failures:
FlashQLA versus original FLA had adapter-gradient relative L2 difference
0.57623942; the broader eight-example adaptive partition comparison had
0.17602324. Both were finite and accepted by the previously selected finite
policy. Those receipts do not establish strict numerical parity. Neither these
acceptances nor the packing checks establish general gradient agreement or
model quality.

Runtime receipts are under
`results/student_injection_awareness/4b/regular/causal_adapter/`, including
`gated_delta_canary.json`, `adaptive_gradient_canary.json` and
`packing_canary.json`. Training logs are under
`logs/runpod/student_injection_awareness/`. The authorized network volume was
expanded from 100 to 200 GB to accommodate the paired adapters and pinned 9B
base; the existing compute instance was retained.

## Evaluation input audit

Each instruction was audited on all 3,012 ID and 6,395 OOD inputs before model
evaluation. Regular maximum lengths were 29,513 and 29,570 tokens; aware
maximum lengths were 29,714 and 29,771. All fit the 32,768-token serving context
with zero truncation. The pinned 4B and 9B tokenizer JSON, tokenizer
configuration and chat template have identical SHA-256 hashes, so this rendering
audit applies to both sizes. Receipts are
`results/student_injection_awareness/evaluation_token_audit.json` and
`tokenizer_identity_audit.json`; initialization inspection is recorded in
`initialization_audit.json` in the same directory. No held-out model scores
were used for these length checks.

## Regular 4B training completed

The regular condition completed the fixed one epoch: 272 optimizer updates,
all 8,688 examples and 83,816,369 direct tokens, with zero padding. Recorded
training runtime was 3,916.06 seconds. Excluding the first two updates, mean
update time was 12.49 seconds, median 12.33 and p90 15.41; this exclusion does
not remove every compilation event. Peak allocated GPU memory was 150.02 GiB.
Mean training loss was 0.23312; the first five-update report was 0.70117 and
the last five-update report at step 270 was 0.19723. These are training reports
on different batches, not held-out quality evidence.

The final FP32 master tensor digest is
`871446f2bb31e973f0bbf3ec2edc7e1dcbe4114b774e79be6e0b4c68b1d94eab`,
different from initialization. All 256 tensors (169,869,312 elements) are finite
FP32, and the serving export retains their exact values under rebased names.
Master file SHA-256 is
`ef2ed374f7f07dfcf1455fc3c78a7722a7ce128dac0263deb79e8cc27560b873`;
serving file SHA-256 is
`2d6a96428ff904b6254f7b20924650e6c4da2b93046673b6937fc76189029924`.
The teacher-target file hash remains
`1ae8c3cccc2546335f8002d1475cd86d7a7e059fedb66345d1aa13d6a30a526a`.
The completed master, export and receipts were collected locally; artifact
verification is in `results/student_injection_awareness/4b/regular/artifact_integrity.json`.
Serving score parity, ID/OOD evaluation and publication are still pending.

## Injection-aware 4B startup

The second condition audited all 8,688 inputs: 85,562,657 tokens, maximum
29,538, zero truncation. Its initial master tensor digest matches the regular
condition exactly. Strict eager and compiled packing checks passed, with
gradient relative L2 differences 0.00745419 and 0.00823637. The longest-input
preflight passed on 32 examples totaling 928,943 tokens, peaking at
161,086,033,920 allocated bytes, before optimizer updates began.

The separate FlashQLA/FLA and adaptive partition comparisons again failed
strict parity (relative L2 differences 0.57791144 and 0.17648131), were finite,
and used the existing selected finite acceptance policy. Failed strict results
remain explicit in their receipts. Teacher targets are unchanged. These
startup checks do not establish held-out quality.

## Injection-aware 4B training completed

The injection-aware condition completed the same fixed epoch and 272 updates,
covering all 8,688 examples and 85,562,657 direct tokens with zero padding.
It used 4,616 physical calls, compared with 4,550 for the regular instruction.
Recorded training runtime was 3,471.49 seconds. Excluding the first two updates,
mean update time was 12.54 seconds, median 12.65 and p90 15.49. Peak allocated
GPU memory was 150.98 GiB. Mean training loss was 0.23312 and the final
five-update report at step 270 was 0.19855. Neither the similar mean training
losses nor these batch reports establish an ID/OOD effect of the instruction.

Both conditions have the same initial master tensor digest and teacher-target
file hash. The aware final tensor digest is
`6d72b20a2f7e756410fcb20f0818915963d016021cb74c4fda565c24ab9cb09a`.
All 256 tensors (169,869,312 elements) are finite FP32 and exactly equal to
their rebased serving counterparts. Master file SHA-256 is
`d880686818a56ece8efedbd5dca9eb4a8067dd468686c5849024c307466e7a4f`;
serving file SHA-256 is
`999182e8da1706c26a0bcc8bad2cde691c28fa4ad586ad2df0ba146a42de326b`.
The remote verification receipt is
`results/student_injection_awareness/4b/injection_aware/artifact_integrity.json`.
The completed adapters and receipts were collected locally, and the local
integrity check passed. Bounded eager reference scoring has started.
Serving parity, complete ID/OOD scores and publication remain pending.

## Serving startup recovery

Bounded eager reference scoring completed on the fixed 20 training examples
per instruction using original pinned FLA kernels. All scores were finite;
maximum adapter effects were 0.88874 for regular and 0.87298 for aware. These
checks establish an adapter effect on the parity cohort, not held-out quality.

The first vLLM startup failed before any evaluation rows were produced. Its
copied compiler artifacts referenced missing files beneath `/root/.cache/vllm/`
while the current cache root was on the network volume. The failure log is
preserved as `logs/runpod/student_injection_awareness/eval_vllm4b.failed_stale_cache.log`.
The retry uses fresh campaign directories for `VLLM_CACHE_ROOT` and
`TORCHINDUCTOR_CACHE_DIR`, preserving the model, engine settings, parity gates
and evaluation inputs. Its recovery receipt is
`results/student_injection_awareness/4b/serving_cache_recovery.json`.
The retry has started; serving parity and complete quality results remain pending.

## Regular 4B ID evaluation completed

The fresh-cache vLLM engine passed the regular adapter's serving gate. On the
20-example training cohort, adapter correlation with the FP32 causal master was
0.999824 and mean absolute score difference was 0.003586; base-model values were
0.999553 and 0.004753. Maximum adapter effects were 0.888744 in eager inference
and 0.898995 in vLLM. This establishes the recorded bounded serving agreement,
not general gradient parity.

All 3,012 canonical ID examples were scored: 946 test STRIDE and 2,066 Gloom
exfiltration. Local verification checked the complete ID membership, labels,
sources, prompt hashes, configuration identity and finite scores. Actual input
tokens matched the pre-evaluation audit exactly: 33,750,959 total and 29,513
maximum, with no truncation. The measured evaluation loop took 658.27 seconds;
this excludes initial prompt preparation and engine startup.

| ID source | Rows | pAUROC@20 | AUROC |
| --- | ---: | ---: | ---: |
| Gloom exfiltration | 2,066 | 0.763290 | 0.918684 |
| Test STRIDE | 946 | 0.929255 | 0.984104 |
| Source macro | 3,012 | 0.846273 | 0.951394 |
| Pooled | 3,012 | 0.816665 | 0.942615 |

At the fixed 0.5 threshold, source-macro recall was 0.830638 and FPR 0.051326.
Source-macro Brier score was 0.089216 and pooled Brier 0.097827. Pooled ECE
was 0.058943 and log loss 0.318370. There were 1,569 unique scores and 1,443
tied rows. The complete calibration bins and
threshold diagnostics are in
`results/student_injection_awareness/4b/regular/id/result.json`; predictions
and this result were collected locally. Regular OOD evaluation is advancing
in the same persistent engine. The awareness intervention's matched ID/OOD
comparison and publication remain pending.

## Regular 4B OOD evaluation completed

All 6,395 frozen OOD examples were scored. Local verification checked complete
membership, labels, sources, prompt hashes, configuration identity and finite
scores. Actual input lengths again matched the independent audit: 55,346,879
tokens total, maximum 29,570, with zero truncation. The measured evaluation
loop took 991.04 seconds, excluding initial prompt preparation and engine
startup. The result, predictions and local coverage receipt are under
`results/student_injection_awareness/4b/regular/ood/`.

| OOD source | Rows | pAUROC@20 | AUROC |
| --- | ---: | ---: | ---: |
| AgentDojo | 1,500 | 0.676676 | 0.920209 |
| CUA Shade Arena | 955 | 0.777495 | 0.895061 |
| IAC | 1,443 | 0.916043 | 0.979077 |
| MLE sabotage | 148 | 1.000000 | 1.000000 |
| Rogue Eval | 1,499 | 0.947882 | 0.986700 |
| Shade Arena | 850 | 0.339330 | 0.705038 |
| Source macro | 6,395 | 0.776238 | 0.914348 |
| Pooled | 6,395 | 0.624621 | 0.898344 |

At the fixed 0.5 threshold, source-macro recall was 0.721999 and FPR 0.094816.
AgentDojo FPR was 0.400000; Shade Arena recall was 0.428230. The source spread
limits any claim based only on the aggregate. Source-macro Brier was 0.138607;
pooled Brier was 0.147491, ECE 0.099895 and log loss 0.455545. There were
1,927 unique scores and 4,468 tied rows. Full per-source calibration bins and
threshold diagnostics remain in the result JSON.

The injection-aware adapter also passed serving parity: adapter correlation
0.999759 and mean absolute score difference 0.003326; base values 0.999170
and 0.007212. Both backends had a maximum adapter effect of 0.872977. Its ID
evaluation is advancing in the same vLLM engine. The regular baseline is
complete; a matched awareness comparison and publication remain pending.

## Matched 4B ID comparison

The aware condition also completed all 3,012 ID examples. Local checks verified
complete membership, labels, sources, prompt hashes and configuration identity;
all scores were finite. Input lengths matched the independent audit exactly:
34,356,371 tokens total, maximum 29,714, with zero truncation. Its evaluation
loop took 668.10 seconds, excluding initial prompt preparation and startup.
The result, predictions and local coverage receipt were collected under
`results/student_injection_awareness/4b/injection_aware/id/`.

| ID view | Regular pAUROC@20 | Aware pAUROC@20 | Aware minus regular | Regular AUROC | Aware AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gloom exfiltration | 0.763290 | 0.753582 | -0.009708 | 0.918684 | 0.909249 |
| Test STRIDE | 0.929255 | 0.941021 | +0.011765 | 0.984104 | 0.986722 |
| Source macro | 0.846273 | 0.847301 | +0.001029 | 0.951394 | 0.947986 |
| Pooled | 0.816665 | 0.816798 | +0.000133 | 0.942615 | 0.940485 |

The primary ID score changed little, with opposite source effects. Macro
AUROC decreased by 0.003408. At the unchanged 0.5 threshold, macro FPR fell
from 0.051326 to 0.032391 while recall fell from 0.830638 to 0.781863.
Macro Brier worsened from 0.089216 to 0.099054; pooled ECE increased from
0.058943 to 0.098971 and log loss from 0.318370 to 0.355671. The aware scores
had 1,483 unique values and 1,529 tied rows.

This single-seed comparison changes the instruction in both training and
evaluation and does not isolate learned resistance from test-time elicitation.
It does not establish a uniform ID benefit. Aware OOD evaluation is advancing;
the complete ID/OOD comparison, uploads and 9B campaign remain pending.

## Matched 4B OOD comparison completed

Both 4B conditions now have complete ID and OOD results. The aware condition
scored all 6,395 OOD examples. Local verification checked complete membership,
labels, sources, prompt hashes, configuration identity and finite scores.
Lengths matched the independent audit: 56,632,274 tokens total, maximum 29,771,
with zero truncation. The measured evaluation loop took 999.86 seconds,
excluding initial preparation and engine startup. All four evaluations used
one persistent vLLM engine after the documented cache recovery; it exited
normally after the last result. Predictions, results, adapters and runtime logs
have been collected locally.

| OOD view | Regular pAUROC@20 | Aware pAUROC@20 | Aware minus regular | Regular AUROC | Aware AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| AgentDojo | 0.676676 | 0.824868 | +0.148191 | 0.920209 | 0.955834 |
| CUA Shade Arena | 0.777495 | 0.777092 | -0.000404 | 0.895061 | 0.908190 |
| IAC | 0.916043 | 0.910398 | -0.005644 | 0.979077 | 0.977892 |
| MLE sabotage | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1.000000 |
| Rogue Eval | 0.947882 | 0.940592 | -0.007290 | 0.986700 | 0.984058 |
| Shade Arena | 0.339330 | 0.198990 | -0.140340 | 0.705038 | 0.632088 |
| Source macro | 0.776238 | 0.775323 | -0.000914 | 0.914348 | 0.909677 |
| Pooled | 0.624621 | 0.638546 | +0.013925 | 0.898344 | 0.899471 |

Primary OOD performance changed little: the AgentDojo improvement was offset
by deterioration on Shade Arena and smaller decreases elsewhere. Macro AUROC
fell by 0.004670. At the unchanged 0.5 threshold, macro FPR fell from 0.094816
to 0.068390 while recall fell from 0.721999 to 0.632569. Macro Brier worsened
from 0.138607 to 0.161174; pooled ECE increased from 0.099895 to 0.149339 and
log loss from 0.455545 to 0.510851. The aware scores had 1,632 unique values
and 4,763 tied rows. Complete calibration bins and per-source threshold
diagnostics remain in the result JSON.

The paired summary verifies identical initial adapters and cached teacher
targets, both serving parity gates, and full evaluation coverage. These
single-seed results do not establish a general benefit from the explicit
warning. Both conditions are retained for release; no prompt, threshold or
checkpoint was selected using these test results. Publication and the matched
9B campaign remain pending.

## 4B publication and 9B startup

Both 4B adapters were published and verified at the returned remote revision:

- [Regular](https://huggingface.co/Jazhyc/Gleipnir-4B-ToolTrajectories-Regular),
  revision `3ade2aa9fe731cd7d502eb435088e76f9b31d215`.
- [Injection-aware](https://huggingface.co/Jazhyc/Gleipnir-4B-ToolTrajectories-InjectionAware),
  revision `841dd32ce3940078b2a26ce47f84014bb2746019`.

Automatic approval review rejected the initial broader publication payload.
The uploader was narrowed to an explicit six-file allowlist: both adapter
weights and loading configurations, license and model card with the student
instruction. No training metadata, provenance, evaluation reports, parity
receipts or release manifests were uploaded. The narrower upload was approved.
Remote verification checked exact allowed file coverage (apart from generated
`.gitattributes`) and both weight LFS SHA256 values before writing each receipt.
Full research artifacts and upload receipts remain local. Focused publication
tests passed (3 tests); Ruff lint and format checks passed.

The matched 9B pair has started on the existing B200, with pinned cached
`Qwen/Qwen3.5-9B`, fresh seed-0 initialization, all-layer nonreentrant
checkpointing and the supported checkpointed compilation policy. Training
inputs, soft teacher targets, learning rate, full-epoch stopping rule and
evaluation gates are unchanged. Startup gates and training results remain
pending; there are no 9B quality claims yet.

## 9B packed-runtime startup guard recovery

The first 9B attempt completed its input audit (8,688 rows, 83,816,369 tokens,
maximum 29,337, zero truncation), then failed before model loading or optimizer
updates. `validate_packed_training_config` accepted only the uncheckpointed
`full_attention_and_linear_shell` compilation policy, rejecting the trainer's
supported `checkpointed_full_attention_and_linear_shell` route. This was a
startup configuration failure, not an OOM or numerical training failure.

The runtime now accepts that route only when gradient checkpointing and
nonreentrant checkpointing are both explicitly true. BF16/FlashQLA settings,
objective restrictions, eager and compiled packing isolation gates, longest-row
backward preflight and finite-gradient checks are unchanged. This shared fix
applies to both 9B conditions. The full composed 9B launch configuration now
passes the startup guard in a regression test; false checkpointing combinations
remain rejected. All 15 focused local checks passed; Ruff lint/format checks
passed. The remote source hash matches the tested local file.

The failed log is preserved as
`logs/runpod/student_injection_awareness/train_9b.failed_packing_compile_guard.log`.
Its SHA256 is `b5c3ee5a35fbcb7f3fbfe6f04089397a47313dea5b3308e2d2791fba76199ec1`.
The original job, execution contract, token audit and failure receipt are
collected locally under `results/student_injection_awareness/9b/startup_recovery/`.
The corrected matched pair has restarted; model loading and native training
canaries remain pending. No benchmark rows or training settings were reduced.

## Regular 9B native gates and training startup

The retry loaded the pinned causal-LM 9B base and verified 8,953,803,264 frozen
BF16 elements and 232,783,872 trainable FP32 LoRA elements in 256 tensors.
All 24 GDN layers use FlashQLA. Both mandatory packing isolation gates passed:
eager adapter-gradient relative L2 was 0.008496 and compiled was 0.007835,
within the unchanged 0.05 tolerance. The longest-32 backward preflight passed
with 922,511 actual tokens, maximum 29,337, nonzero finite gradient norm
13.216311, unchanged masters and peak allocated memory 34.2383 GiB.
Collected native receipts are under
`results/student_injection_awareness/9b/regular/causal_adapter/`.

The fresh seed-0 initial master digest is
`5ba0b3d6e44c20a1a1ccfd2d5923b6d90a5dd930e02321ff832cb471984bcf79`.
It remains to be matched against the aware condition's actual initialization.
The separate FlashQLA-versus-FLA strict gradient result failed (relative L2
0.226489), as did the broader adaptive-partition comparison (0.307071).
Both were finite and accepted only by the existing explicit `selected_finite`
policy; these negative strict results are preserved. They do not alter the
mandatory packing gates or establish general gradient parity.

The regular condition has completed its first optimizer updates. The first
update took 30.87 seconds, including startup costs; warmed full-epoch timing,
training completion, paired initialization, serving parity and quality results
remain pending. GPU ECC checks showed zero volatile uncorrected errors.

## Regular 9B training completed

The regular condition completed all 272 optimizer updates and one full epoch
over the unchanged 8,688 trajectories. Packed execution consumed 83,816,369
tokens in 4,550 physical forwards, with zero padding or truncation. Training
runtime was 6,229.54 seconds; after excluding the first two updates, mean update
time was 22.69 seconds (median 22.54, p90 28.16). Mean training loss was
0.222823. Peak allocated memory was 35.7395 GiB. All reported losses and
gradient norms were finite; volatile uncorrected GPU ECC remained zero.

The final master tensor digest is
`41c6b4bbb8331e76a4f5dbec3f5a1c270524017178e8aa811d36b05bdf196dc1`,
different from the recorded seed-0 initialization. A CPU artifact audit checked
all 256 tensors (232,783,872 elements) for FP32 storage and finite values,
verified both file hashes against the rebase manifest, and confirmed exact
tensor equality after rebasing the adapter for serving. The master file SHA256
is `5976ef206a8ed2d66cd536c7b14d3eb3d97d14e6b168cae16dac410a1fd6501f`;
the serving file SHA256 is
`3dfc9a9ca9b96c3126cb53c0f984728eaf7377de598656c34ea38b326e587df5`.
The audit also verified the complete update count and unchanged teacher cache
hash. Receipts remain under `results/student_injection_awareness/9b/regular/`.

The aware condition started automatically in the same launch chain, completed
its input audit and started native initialization. Its startup gates, paired initialization,
training completion and both conditions' serving parity and quality results
remain pending. The separate failed strict gradient comparisons recorded above
remain negative results; the artifact checks do not establish serving parity.

## Aware 9B native gates and paired initialization

The aware condition audited all 8,688 rows (85,562,657 tokens, maximum 29,538,
zero truncation) and verified the same BF16 base and 232,783,872 FP32 trainable
elements as regular 9B. Its actual initial master digest exactly matches the
regular digest recorded above. Mandatory eager and compiled packing gates both
passed, with adapter-gradient relative L2 of 0.008496 and 0.007835 respectively.
The longest-32 backward preflight passed with 928,943 actual tokens, maximum
29,538, finite nonzero gradient norm 12.239272, unchanged masters and peak
allocated memory 34.3467 GiB. These native receipts are collected locally.

The separate strict FlashQLA-versus-FLA gradient comparison failed (relative L2
0.226668); the broader adaptive-partition comparison also failed (0.303002).
Both were finite and accepted only under the unchanged explicit
`selected_finite` policy. The negative results are retained independently of
the mandatory packing gates. The first optimizer updates are advancing; full
training completion, serving parity and full benchmark results remain pending.

## Matched 9B training completed

Both conditions completed all 272 optimizer updates and one full epoch over
the unchanged 8,688 trajectories. The aware condition consumed 85,562,657
tokens in 4,616 physical forwards, with zero padding or truncation. Its training
runtime was 6,317.56 seconds (1h 45m 18s), compared with 6,229.54 seconds
(1h 43m 50s) for regular 9B. Excluding the first two aware updates, mean update
time was 23.02 seconds (median 22.89, p90 28.43). Mean training loss was
0.224380 and peak allocated memory was 35.8480 GiB. Reported losses and gradient
norms were finite throughout both runs.

The aware final master tensor digest is
`f0edb8589bb6e841d85b88a56ce20fd7a3ffbce5fee1424ccfee5b6fa4d046eb`,
different from the verified common initialization. The artifact audit checked
all 256 tensors (232,783,872 elements) for finite FP32 values and exact serving
rebase equality, verified both hashes against the rebase manifest, and checked
the full update count, row/token coverage and unchanged teacher-cache hash.
The aware master file SHA256 is
`c12a313b759d5af39c015c1bc705bf6b1051984b1cfe32287eea080f324e0a75`;
its serving file SHA256 is
`8ce27cefe83079f5d9f4d7c038ffb6b7dfb45561c4cb1759e7c3bc0216320cd8`.
Both 9B artifact audits passed. The full training log is collected locally.

A GPU health snapshot during aware training is collected under
`results/student_injection_awareness/9b/gpu_health_aware_training.json`.
It records zero volatile and aggregate corrected/uncorrected DRAM and SRAM
ECC errors, no pending repairs and recovery action `None`; GPU and memory
temperatures were 56 and 51 degrees C. Retired-page values were unavailable,
and row-remapper data was absent; no zero-count claim is made for those fields.

The bounded original-FLA reference canaries have started on the existing B200.
Serving score parity, full ID/OOD evaluation and 9B publication remain pending.
The separate failed strict gradient comparisons remain negative results under
the explicit `selected_finite` policy; training completion does not establish
general kernel gradient parity or monitor quality.

## 9B serving canaries and regular ID completed

Both bounded original-FLA reference evaluations completed on the same 20
training examples per condition. Scores were finite, reference master hashes
matched the audited checkpoints, and adapter effects were nonzero (maximum
absolute score changes 0.881815 regular and 0.750761 aware). Regular 9B then
passed the unchanged vLLM serving gate: adapter score correlation 0.999686,
mean absolute difference 0.005229; base correlation 0.999804 and mean absolute
difference 0.003914. Its vLLM adapter effect was 0.881815. Aware serving parity
remains pending until the regular full evaluations finish.

Regular 9B completed all 3,012 canonical ID examples in 899.81 seconds of
evaluation-loop runtime, excluding engine initialization and input preparation.
An independent local audit verified exact unique membership, labels, sources,
configuration and prompt hashes, finite digit logprobs and score extraction,
and per-example token encoding against the verified identical 4B tokenizer.
All 33,750,959 input tokens were retained; the longest prompt was 29,513 tokens.
The prediction SHA256 is
`05c72d949ca38b043c0aaceb16ac0bd70f179313ebda49d907dde8826779c4f6`.

The primary equal-source macro pAUROC@20 is 0.878686 (regular 4B: 0.846273),
macro AUROC 0.960759 and Brier 0.078493. At the fixed 0.5 threshold, macro
recall is 0.861747 and FPR 0.062236. Source pAUROC@20 values are 0.809781
for Gloom exfiltration and 0.947591 for STRIDE. Pooled pAUROC@20 is 0.846824,
AUROC 0.952068, Brier 0.085485, ECE 0.037719 and log loss 0.284001. There
are 1,633 unique scores and 1,379 tied rows; source calibration bins and
threshold diagnostics are retained in the full local report.

Regular OOD and both aware full evaluations remain pending. This partial
milestone does not establish the paired 9B prompt effect or OOD performance;
no checkpoint or threshold is selected using these results.

## 9B FlashInfer serving stall and consistent backend recovery

The first 9B vLLM attempt stalled during regular OOD after saving 1,920/6,395
predictions. Within the next 128-row batch, 95 requests completed before
progress stopped. Repeated observations confirmed an unchanged log for over
four minutes despite a live worker: GPU utilization 100%, memory utilization
0%, power about 237 W and temperature 32 degrees C. Volatile uncorrected ECC
remained zero. The exact stuck kernel is unknown: `py-spy` attachment was denied
by the container's ptrace permissions. This is distinct from the earlier
compiler-cache relocation failure.

The failed log SHA256 is
`834de093026e2d169f55fd7a6c7716812b24357b605fad47d5f2872f7406fb1b`.
The log, telemetry, complete first-attempt regular ID result and partial OOD
predictions are retained under
`results/student_injection_awareness/9b/runtime_recovery/flashinfer_stall/`.
The failed worker was explicitly stopped and GPU process release verified;
the container does not support psutil's pidfd-based wait, so an explicit
same-worker kill and GPU process check were used after graceful termination.

An [upstream FlashInfer GDN hang report](https://github.com/flashinfer-ai/flashinfer/issues/3329)
describes a similar spin-wait signature and a Triton prefill workaround, but
uses different hardware, shapes and MTP. It motivates a recovery candidate;
it does not identify our failure's cause. The replacement persistent vLLM run
uses Triton GDN prefill for both 9B conditions. All four 9B suites will be rerun
from the beginning after the unchanged original-FLA/master serving gates.
The first-attempt ID metrics above are archived historical results pending
the consistent replacement evaluation.

The evaluator records the explicit GDN backend in parity and runtime receipts
and gives a backend override a distinct prediction contract, preventing mixed
cache reuse. Default FlashInfer identities, including the completed 4B results,
are preserved. Inputs, prompt hashes, BF16 base weights, FP32 adapters, batching,
one-token scoring and parity thresholds remain frozen. Training is complete
and unaffected. The replacement engine loaded, compiled and warmed up, then
regular 9B passed serving parity: adapter correlation 0.999784 and mean absolute
difference 0.003099; base correlation 0.999665 and mean absolute difference
0.005366. Its served maximum adapter effect was 0.871565. The 13 focused
contract/publication tests passed, including invalidation of cached predictions
when the backend changes, and Ruff passed. Full replacement evaluations and
aware serving parity remain pending.

## Replacement 9B regular ID completed

The Triton-backed replacement completed all 3,012 regular ID examples in
1,036.80 seconds of evaluation-loop runtime. Its independent coverage audit
passed exact unique membership, labels/sources, prompt/config identity, finite
digit logprobs, score extraction and per-example token encoding. All 33,750,959
input tokens were retained, with a longest prompt of 29,513. Prediction SHA256:
`3be8cc4ef7fef9e3646c5c7397f7ef7bc495edccdb42cf5c12a1857bc4bf02de`.

Primary equal-source macro pAUROC@20 is 0.879559, macro AUROC 0.960917 and
Brier 0.078196. At the fixed 0.5 threshold, macro recall is 0.862232 and FPR
0.061753. Source pAUROC@20 values are 0.811737 for Gloom exfiltration and
0.947380 for STRIDE. Pooled pAUROC@20 is 0.848036, AUROC 0.952174, Brier
0.085105, ECE 0.036282 and log loss 0.283268. There are 1,645 unique scores
and 1,367 tied rows. Calibration bins and source threshold diagnostics remain
in the full local report.

These values supersede the archived first-attempt regular ID result for the
paired 9B comparison. The backend changed solely to recover the observed
runtime stall, before inspecting replacement quality. Regular OOD, aware parity,
both aware full suites and 9B publication remain pending.

## Full regular 9B evaluation and aware serving parity completed

Regular 9B completed all 6,395 strict OOD examples in 1,523.34 seconds of
evaluation-loop runtime. The local audit passed complete unique membership,
labels/sources, prompt and backend-specific configuration hashes, finite
logprobs and extracted probabilities, and every example's token encoding.
All 55,346,879 prompt tokens were retained, with maximum length 29,570. OOD
prediction SHA256:
`bc1a3e32f0c2765659cb683c7b92e4975b25be6fd5cba827fef0a922af5a13e8`.
Both regular full-suite coverage receipts passed and are collected locally.

Primary equal-source macro OOD pAUROC@20 is 0.837368, AUROC 0.926803 and
Brier 0.129929. At threshold 0.5, macro recall is 0.708062 and FPR 0.047912.
Pooled pAUROC@20 is 0.740115, AUROC 0.928193, Brier 0.134148, ECE 0.128152
and log loss 0.424799. There are 2,263 unique scores and 4,132 tied rows.

| OOD source | Rows | pAUROC@20 | AUROC | Brier | Recall at 0.5 | FPR at 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AgentDojo | 1,500 | 0.918773 | 0.982399 | 0.061657 | 0.953333 | 0.101333 |
| CUA Shade Arena | 955 | 0.813916 | 0.932872 | 0.153357 | 0.632353 | 0.012526 |
| IAC | 1,443 | 0.930296 | 0.984580 | 0.153316 | 0.579021 | 0.000000 |
| MLE sabotage | 148 | 0.998980 | 0.999796 | 0.040470 | 0.928571 | 0.000000 |
| Rogue Eval | 1,499 | 0.965763 | 0.991145 | 0.117959 | 0.674232 | 0.000000 |
| Shade Arena | 850 | 0.396482 | 0.670028 | 0.252814 | 0.480861 | 0.173611 |

Regular 9B exceeds regular 4B's primary macro pAUROC on these frozen suites,
but source outcomes and calibration are uneven. For example, OOD pooled ECE
is worse, and Shade Arena AUROC remains low. Cross-size comparisons also use
different serving GDN backends after the documented recovery, limiting pure
backbone-size attribution. No promotion or threshold retuning follows from this
single-seed result. The matched 9B prompt comparison remains pending.

Aware 9B passed the unchanged Triton serving gate: adapter correlation
0.999804 and mean absolute difference 0.003316; base correlation 0.999716 and
mean absolute difference 0.004490. The maximum served adapter effect is
0.750761. Its full ID/OOD evaluations are now queued in the same persistent
engine. The regular upload has started using the six-file public allowlist;
the publisher's optional single-condition selection passed all seven mocked
publication tests and Ruff.

Regular 9B is publicly published at
[Jazhyc/Gleipnir-9B-ToolTrajectories-Regular](https://huggingface.co/Jazhyc/Gleipnir-9B-ToolTrajectories-Regular),
revision `536efd527adc283aa8c237a0f58808079aa7ed02`. An independent remote read
verified the actual public head, the exact six allowed files plus Hub
`.gitattributes`, both adapter LFS checksums and downloaded small-file hashes
against the local manifest. Private provenance, training/runtime metadata,
evaluation/parity reports and manifests are absent from the public repository.
The local release-manifest SHA256 is
`b2d27b80f3e3c255ffb7d1041c28c08f7c42ee9977eb5514caa361e2074e2a12`.
The independent receipt is saved in `9b/publication_integrity.json`; it currently
covers regular only. Aware full evaluation and publication remain pending.

## Matched 9B ID comparison completed

Aware 9B completed all 3,012 canonical ID examples in 1,056.77 seconds of
evaluation-loop runtime. The full local coverage audit passed unique membership,
labels/sources, instruction/configuration hashes, finite digit logprobs,
probability extraction and per-example token encoding. All 34,356,371 input
tokens were retained; the longest prompt was 29,714 tokens. Prediction SHA256:
`d8df4afb53baa8cc62ebaf8f0fa4e0205e1dd947b41768b46d19c5df35ee431a`.

| Equal-source macro ID metric | Regular 9B | Aware 9B | Aware minus regular |
| --- | ---: | ---: | ---: |
| pAUROC@20 (primary) | 0.879559 | 0.896654 | +0.017096 |
| AUROC | 0.960917 | 0.968610 | +0.007694 |
| Brier | 0.078196 | 0.074435 | -0.003761 |
| Recall at 0.5 | 0.862232 | 0.875668 | +0.013436 |
| FPR at 0.5 | 0.061753 | 0.057311 | -0.004442 |

Gloom exfiltration pAUROC@20 increased from 0.811737 to 0.849322; STRIDE
decreased from 0.947380 to 0.943987. Aware pooled pAUROC@20 is 0.866151,
AUROC 0.959555, Brier 0.078020, ECE 0.035885 and log loss 0.265456. Its
1,569 unique scores leave 1,443 tied rows. Source calibration bins and
threshold diagnostics are preserved in the full report; the source effects
are mixed even though the macro ID result improves.

The final aware OOD suite is preparing in the same persistent Triton-backed
vLLM engine. No OOD conclusion or uniform prompt benefit follows from this
ID-only comparison. This remains one seed and a joint training/evaluation
instruction intervention with unchanged targets and paired initialization.
