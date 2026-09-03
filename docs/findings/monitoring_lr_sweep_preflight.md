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

The v2 recipe uses microbatch 4 with accumulation 8. This preserves effective
batch 32, one epoch, optimizer steps, scheduler, objective, and every statistical
comparison in the sweep. It must pass the same longest-sequence preflight before
training begins. The failed v1 runtime status and full log are retained in the
ignored result and log trees as `preflight_mb8_failure_status.json` and
`preflight_mb8_failure.log`.
