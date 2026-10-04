# Monitor-directed injection augmentation at 9B

Date: 2026-10-04. Status: training advancing; evaluation queued. No new
9B quality result is available yet.

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

User-authorized shutdown remains pending until complete evaluation, artifact
collection and verification. There is no separate in-chat heartbeat scheduler;
active goal execution can inspect progress but a remote queue is not an agent
follow-up. The goal remains active through verified shutdown.

The source-checked launch started PID 15742 on B200 `3422tdq7tfog52`; training
worker PID 15745 is live. Model loading verified 8,953,803,264 frozen BF16
elements and 232,783,872 trainable FP32 elements in 256 tensors. Live all-layer
nonreentrant checkpointing matches the recorded recipe. All 24 GDN layers bound
to FlashQLA with startup diagnostics explicitly marked reused/not repeated.
The populated shared compiler cache resolves to
`/workspace/gleipnir/.cache/training/student_injection_awareness`.
Preparation manifest SHA256 is
`8ea9596a8458520525747c31b19eb248549a510b087e5ffd45c94baf4fce4442`.
Launch and cache receipts are collected locally; serving parity and all fresh
evaluation outputs remain pending. An initial launcher import-path failure
started no workload and is preserved separately; the source-checked launch
uses an explicit repository import path.

Ordinary training reached at least 2/272 completed updates, with mandatory
finite/missing-gradient guards permitting both updates. First-update elapsed
time was 282.16 seconds including compilation; the next update took about
20 seconds. GPU utilization reached 100%, with 37,470 MiB used. These early
timings do not establish whole-epoch throughput. The numerical training command
matches the original 9B regular condition; only campaign paths and the explicit
startup-validation reference differ. Feature implementation is committed as
`ac21e87`; the full goal remains active until verified shutdown.
