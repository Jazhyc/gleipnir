# Monitoring-only 4B LR sweep preflight

## 2026-09-03: microbatch 8 is not safe at the frozen length ceiling

Before launching the learning-rate screen, the v1 recipe tested rank-128 QLoRA
with microbatch 8 and accumulation 4 on the 32 longest monitoring examples. The
longest selected direct context was 29,337 tokens against a configured maximum of
29,696. FLA 0.5.2 bound through `fla.ops.gated_delta_rule.chunk`, and
causal-conv1d 1.6.2.post1 bound through
`causal_conv1d.causal_conv1d_interface`.

The first training step OOMed on an H100 SXM5 80 GB. PyTorch reported 78.09 GiB
in use (74.67 GiB allocated), 1.08 GiB free, and a failed request for another
3.54 GiB. No learning-rate cell started. The frozen v1 stop condition therefore
worked as intended.

The v2 recipe used microbatch 4 with accumulation 8, but it also OOMed on the
same preflight: 77.58 GiB was in use, 1.59 GiB was free, and a further 1.99 GiB
allocation failed. No learning-rate cell started.

The v3 recipe therefore used microbatch 2 with accumulation 16. It cleared the
memory bottleneck, but the first backward pass failed while compiling FLA's
optional TileLang `chunk_bwd_dqkwg` implementation: TVM layout inference reported
conflicting layouts for `b_dq`. This is a backend compiler failure rather than an
OOM.

FLA 0.5.2's dispatcher explicitly supports `FLA_DISABLE_BACKEND_DISPATCH=1`,
which bypasses optional backends and uses the operation's default Triton
implementation. Version 4 freezes that setting while retaining microbatch 2 and
accumulation 16. The training metadata records the setting, and the campaign
validator requires it. Effective batch 32, epoch count, optimizer steps,
scheduler, objective, and every statistical comparison remain unchanged. Version
4 then encountered FLA's Hopper correctness guard because the base environment's
Triton was older than 3.7.1.

Version 5 adds an isolated Triton 3.7.1 target. A matched microbatch-2,
accumulation-16 preflight completed the full optimizer step, adapter save, and
serving rebase in 458 seconds. Both required kernel families remained bound; an
active-step snapshot showed about 68.6 GiB used. Version 5 records and validates
the Triton version for every sweep cell. Failed runtime statuses and logs are
retained in the ignored artifact trees.

## 2026-09-04: stop the sweep for a matched throughput screen

The first two cells were deliberately stopped without checkpoints after the two
lanes reached 73 and 71 of 272 optimizer steps. Both GPUs were healthy and both
kernel families were bound, but the steady rate was about 92 seconds per step,
projecting roughly seven hours per cell. That is slower than the historical
microbatch-1 full-data run despite causal-conv1d.

The microbatch-2 choice was memory-safe but had not been compared against
microbatch 1 on representative mixed-length batches. The collator right-pads
each pair, so padding amplification is the leading hypothesis. Preserve this as
a negative systems result and run the matched 20-step
`monitoring_training_throughput` screen before restarting learning-rate work.
Do not enable naive packing: the hybrid attention/recurrent backbone needs
explicit state resets and multi-boundary selected-logit handling first.
