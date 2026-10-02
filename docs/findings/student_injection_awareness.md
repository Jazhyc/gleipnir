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
