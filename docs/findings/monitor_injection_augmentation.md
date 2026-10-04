# Monitor-directed injection replacement campaign

Date: 2026-10-04. Status: startup verified; training running, evaluation queued.
No training-quality or robustness result is available yet.

The user selected replacing 40% of the 8,688 original regular-prompt training
rows, preserving the original source behavior verdicts and cached Kimi soft
targets. There are 3,475 injected views (1,738 harmless, 1,737 harmful) and 5,213
clean views. Every trajectory is visited once; the matched run has 272 updates,
one epoch, fresh rank-128 initialization and the regular 5e-5 learning rate.
See the [protocol](../../experiments/monitor_injection_augmentation/README.md)
and [decision](../decisions/monitor_injection_augmentation.md).

Eight authored training templates in four families demand either 0 or 1,
independently of the behavior label. The demand agrees in 1,785 cases and
conflicts in 1,690; always reversing it matches only 48.6% of hard labels.
Uniform role quotas are 1,159 user, 1,158 assistant and 1,158 tool messages.
Requested positions follow truncated N(0.5,0.25²) over source characters,
snapped to existing boundaries. All original source bytes and clean JSONL
lines are preserved; four pre-existing XML task-wrapper irregularities remain.
The actual agent continuation is not regenerated, and copied teacher targets
are an explicit off-policy invariance assumption.

The user requested reusing original-adapter results instead of rescoring it.
Fresh evaluation is only for the new adapter, with the regular prompt and
fixed final checkpoint: 5,184 APPS benchmark, 3,930 honest controls, 3,012 clean
ID and 3,012 synthetic held-out ID examples. Synthetic examples use four
templates from two mechanism families excluded from training, with each role
occurring 1,004 times. Every synthetic view has its own clean pair. There is
no original-adapter synthetic baseline, and no APPS-selected checkpoint.
Report agreeing/conflicting verdict demands separately, together with source,
role, family, position, ranking, calibration and clean-calibrated operating points.

Preparation has 83,969,155 input tokens, maximum 29,391, with zero truncation
inside the validated 29,696-token envelope. Training manifest SHA256 is
`e99a700ed1b588323e0f0a757e73e50046496fbc25c29c158b34b129dec99f51`;
held-out evaluation manifest SHA256 is
`88fae967f735ef65fbebc0b25bb820be7f57b6183d2b3859382570fddb1c127b`.
Forty-six focused checks pass locally and on the B200, including resumption,
raw-logprob preservation, rejection of source/label/cache drift, reversible
lowercase/uppercase ID rendering and paired threshold/flip diagnostics.

The source-checked launch uses existing Runpod B200 `3422tdq7tfog52`, PID
13805. Frozen execution code and both materializations match local checksums.
The selected packed BF16 FlashQLA/SDPA recipe, FP32 masters, startup-validation
reuse and persistent compiler cache remain in force. Full serving parity is
queued before vLLM population scoring. Logs are under
`logs/runpod/monitor_injection_augmentation/`; launch, cache and progress
receipts are under `results/monitor_injection_augmentation/`.

The actual Transformers tokenizer audit subsequently completed all 8,688 rows,
with 83,969,213 input tokens, maximum 29,391 and no truncation. Its count is
58 tokens above the raw-tokenizer preparation audit; retain both receipts.
All 15 execution-source hashes match the reviewed local code. Model loading
verified a frozen BF16 base and 256 FP32 trainable LoRA tensors, then explicitly
reused the existing startup validation without repeating numerical probes.
The compiler cache is populated and resolves to the persistent
`/workspace/gleipnir/.cache/training/student_injection_awareness` directory.

Startup subsequently reached at least 6/272 completed updates. The first update
took 268.14 seconds including startup compilation; the next five updates took
approximately 10–15 seconds each (about 12 seconds on average). GPU utilization
reached 100%, with 150,008 MiB used. The reported loss was 0.7229 and gradient
norm 16.14 at step five; mandatory finite/missing-gradient guards permitted the
updates. This is early throughput evidence, not a whole-epoch measurement.
Serving parity and the 15,138 fresh population predictions are still pending.

No in-chat scheduler is available in this session. Active-turn startup checks
do not provide automatic post-turn agent follow-ups; the remote sequential
queue is not claimed as agent monitoring.
