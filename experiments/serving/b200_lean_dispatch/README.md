# Lean dispatch for gleipnir custom ops (not adopted)

`gleipnir.serving.lean_dispatch` registers each `gleipnir::` custom op's
original function directly on the CUDA dispatch key. This bypasses
`torch.library.custom_op`'s `backend_impl`, `_disable_dynamo` wrapper and
aliasing check, which account for about 1 ms of host time per short step in the
`host04` sample. `LeanDispatchWorker` stacks it on the adopted default;
`paired.py` toggles it in one server.

`lean01` (four alternating rounds): C1 scores are bitwise equal in every case,
and the canary is unchanged. Latency changes are within round-to-round noise:
1K p50 29.09 → 28.39 ms (paired −1.6/−0.7/−0.2/−0.7 ms), quick64 mixed sign,
8K slower in 3/4 rounds, full320 c128 throughput ratios 1.015/1.031/0.942/1.008.
Not adopted. Retained as a negative result.
