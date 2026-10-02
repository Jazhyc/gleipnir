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
