# B200 training throughput and attention backend screen

Subsequent decision (2026-10-01): the user selected compiled adaptive batching
for future training after a separate matched execution audit. See the
[recipe decision](../decisions/b200_adaptive_training_recipe.md). The fixed-batch
measurements, selection and ETA below describe this earlier systems screen.

Date: 2026-09-30 / 2026-10-01. Contract:
[`b200_training_throughput`](../../experiments/b200_training_throughput/README.md).
The user authorized short systems benchmarks on the existing B200, leaving the
Pod running. No full training campaign or quality-based model selection occurs.

## Matched workload

The released 4B training population contains 21,837 mixed monitoring/deception
rows with existing Kimi soft targets. A seed-0 dataset/label stratified sample
contains 320 rows: 130 monitoring, 92 Aletheia's Quest, and 98 Liars' Bench.
Ten optimizer updates use effective batch 32. All conditions preserve the
same sample, target, rank, learning rate, and selected-position logit objective.

The base is Qwen/Qwen3.5-4B at
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; rank-128/alpha-256 FP32 adapters,
NF4 double quantization, BF16 compute, AdamW at 5e-5, maximum length 29,696,
FLA/fla-core 0.5.2, causal-conv1d 1.6.2.post1, Triton 3.7.1. Global compilation
is disabled; full-attention layers and linear shells use dynamic Inductor,
keeping the custom recurrent kernels eager. Benchmark adapters are retained
as FP32 masters and are not released monitor checkpoints.

Student rows SHA-256:
`f9b0a322f277991edae2510b658a2298c0c585d402b39f36d6ac86352bf1a76b`.
Soft targets:
`50f163f88b42ff8b6fc04d5ff1334766d6326895ad89c87501a93eeaeae137ee`.
Matched selection:
`a27b8b3d702436839869fda2aa612ad61a4c3ef8aa9f56b4c57a4914b0a50e06`.
Longest-32 preflight selection:
`1ed6774f5234b606a52b19d10a93bf0da9bfbad69b3ffe67af8ef8608e4ed875`.
Legacy deception lengths are inferred solely for selection metadata; original
prompts and labels remain authoritative. Actual direct training input contains
1,314,331 tokens. Grouping changes order and padding and is a batching intervention.

## Cold startup and negative result

The first microbatch-1, accumulation-32 control with checkpoints on all 24
linear layers completed ten updates in 1,481.08 seconds (24m41s). Its first
update took 855.53 seconds (14m16s). Several later updates still compiled or
autotuned; excluding two updates therefore did not isolate steady throughput.
FLA normalization generated hundreds of Triton variants; persisted autotune
results and compatible Inductor/Triton caches substantially reduce repeat cost.

Removing every checkpoint completed one update, then exhausted GPU memory:
176.05 GiB allocated with only about 350 MiB free on the 178.35-GiB device.
This was capacity exhaustion, not allocator fragmentation. The campaign stopped;
the original grouped microbatch-4 condition was never run. Keep this negative
result instead of treating the unexecuted condition as a failure.

Candidates were linked to the control's cache after the startup investigation,
before candidate execution. That intervention is preserved in
`results/b200_training_throughput/runtime_cache_reuse.json`; cold-inclusive
control/candidate timing is confounded. A separate warmed continuation retains
the exact sample and preserves the original failed campaign.

## Warmed checkpoint comparison

The warmed control and half-checkpoint condition both completed ten updates:

| Condition | Trainer loop | Mean update after first two | Peak allocated |
| --- | ---: | ---: | ---: |
| All 24 linear checkpoints, batch 1 x 32 | 367.65 s | 20.24 s | 58.82 GiB |
| 12 linear checkpoints, batch 1 x 32 | 339.03 s | 18.56 s | 131.41 GiB |

Half checkpointing uses layers `[0,2,5,8,10,13,16,18,21,24,26,29]`.
It passed a longest-32 training update at 132.78 GiB allocated and the same-weight
compile canary. The ten-update loop is 7.8% shorter (8.4% higher throughput),
and the mean of the last eight matched random-order updates is 8.3% shorter.
Both produced six Dynamo graphs, finite losses, 320 examples, and zero padding.
This gain trades substantially more memory for less recomputation.

The warmed control's first update still took 180.49 seconds versus 18–22 seconds
thereafter. Model loading, graph reconstruction in a fresh process, saving,
and adapter rebasing are separate from steady GPU compute. Reused disk caches
do not eliminate Python graph capture. Model-quality equivalence is unmeasured.

The first grouped microbatch-2 pass completed in 1,079.71 seconds with eight
Dynamo graphs, 142.51 GiB peak allocation, and 5.22% padding. Individual fast
updates were about 9–10 seconds, but repeated new compilation/autotuning paths
inflated many other updates. Neither its cold-inclusive time nor a subset of
fast updates establishes a warmed gain. A separate matched cached repeat with
its own longest-32 preflight is required before recommending this condition.
That repeat's batch-2 preflight passed: 393.75 seconds for the longest-32 update,
145.44 GiB peak allocation, ten Dynamo graphs, and compile-canary maximum logit
difference 0.0625 with unchanged first-row decision margin. It processed
922,511 actual tokens with 0.045% padding. The representative timing comparison
is still separate from this demanding memory preflight.

The complete cached comparison also finished:

| Condition | Complete Trainer loop | Peak allocated | Dynamo graphs |
| --- | ---: | ---: | ---: |
| Random batch 1 x 32, all checkpoints | 362.09 s | 58.82 GiB | 6 |
| Grouped batch 2 x 16, all checkpoints | 460.08 s | 142.51 GiB | 8 |

Batch 2 is 27.1% longer than its matched control, despite reducing its initial
1,079.71-second loop to 460.08 seconds through cache reuse. Its update times are
`[196.23,9.28,47.48,23.79,57.59,8.90,45.49,9.23,48.49,8.98]` seconds. Mask and
layer-type changes still trigger fresh-process graph capture beyond the first
update. The short grouped updates cannot stand in for the complete workload.
Padding remains 5.22% (1,386,740 padded versus 1,314,331 actual tokens), and the
loss is finite at 0.676149. Reject this batching intervention for the current
recipe. The cached campaign is `results/b200_training_throughput_batching_repeat/`,
source `2fc1b0e4d5e472e50a641098f92b88a66f6b11da`.

## Attention backend interpretation

Original training used Transformers' default PyTorch SDPA, with all CUDA SDPA
backends enabled and no external `flash-attn` package installed. The actual
CUDA SDPA dispatch was not profiled, so describing these runs as external FA2
would be unsupported. FLA 0.5.2 serves the 24 linear-attention layers and is
distinct from FlashAttention. FA4 targets the remaining eight full-attention layers.

The FA4 comparison pins `flash-attn-4==4.0.0b33` in an isolated overlay,
CuTe/CUTLASS 4.8.0, TVM FFI 0.1.14.post1, DLPack 0.1.5, and QuACK 0.6.5.
Both matched attention conditions use the same overlay; locked inference
dependencies remain untouched. Require actual import provenance, CUDA-wheel
integrity, BF16 GQA forward/backward checks, same-weight padded model logits,
compilation parity, and a longest-row training preflight before timing.
The isolated BF16 GQA probe passed on the actual 16-query/4-KV-head, 256-head-dimension
shape: forward relative L2 error 0.001940, dQ 0.002490, dK 0.002477, dV 0.002272
against FP32 math attention. The frozen bounds are 0.02 forward and 0.05 backward.
FA4's opt-in persistent CuTe cache is enabled on the network volume.
The same-weight model check passed at lengths 2,048 and 1,024 with unequal
right padding. SDPA decision logits were `[[8.0625,9.1875],[13.0,10.375]]`;
FA4 returned `[[8.125,9.25],[12.875,10.3125]]`. Maximum logit error is 0.125;
maximum decision-margin error across the two rows is 0.0625. Both rows satisfy
the frozen 0.05 absolute plus 0.01 relative logit tolerance. The trainer's
legacy scalar margin diagnostic refers to the first row only.

The initial FA4 preflight completed its longest-row update in 198.99 seconds,
with 61.80 GiB peak allocation and active `flash_attention_4`. It passed backend
and compile numerical canaries but produced 32 Dynamo graphs, exceeding the
frozen limit of 24. The screen failed before either matched timing condition
started. Logs identify specialization on static `module.layer_idx`.
A separately frozen follow-up enables `allow_unspec_int_on_nn_module` for both
backends, retaining the graph limit and all numerical gates. That follow-up
completed its update but generated 44 graphs: static-index messages disappeared,
while tracing entered FA4's Python implementation. It also failed before either
timing condition started. Keep both failed campaigns intact.
A further matched screen keeps the attention interface opaque to Torch tracing
for both backends, while preserving compiled surrounding layers and the native
GPU kernels; dynamic module integers return to their original setting. The
24-graph limit and numerical/memory gates remain unchanged.
This interface preflight passed with 14 graphs, 61.80 GiB allocated, a 185.55-second
longest-row training loop, unchanged same-weight backend logits, and zero
compile-canary logit difference. Metadata records the original Transformers
interface callable and the explicit eager-interface setting. Both matched
attention timing conditions use this boundary.

Both matched attention conditions completed all ten updates with eight Dynamo
graphs. Their results do not support switching the training backend:

| Backend, opaque interface | Complete Trainer loop | Mean update after first two | Peak allocated |
| --- | ---: | ---: | ---: |
| SDPA | 366.58 s | 20.10 s | 63.43 GiB |
| FA4 4.0.0b33 | 365.80 s | 20.67 s | 61.89 GiB |

The complete-loop time difference is only 0.21%, below the frozen 5% gain gate;
FA4's mean of the last eight updates is 2.85% slower. First updates took 181.25
and 174.43 seconds respectively, so the tiny whole-loop advantage does not
establish a steady compute gain. Losses are finite (0.694015 SDPA, 0.694688 FA4),
but these ten-update losses are not evidence of model-quality equivalence.
Both see the same 320 examples and 1,314,331 tokens without padding. Keep SDPA.
No additional FA4 repeat is required because this candidate is not promising
under the predefined gain rule. No combined FA4/batching recipe is measured.

Integration obstacles were dependency/import isolation for CuTe 4.8.0, unpacking
the native kernel probe's `(output, logsumexp)` return, and excessive Torch
specialization when tracing FA4's Python interface. They were resolved without
relaxing the graph or numerical limits. Native forward/backward support for the
B200 passed; the final negative result concerns speed on this mixed-attention
QLoRA workload, not kernel availability.

The successful attention campaign is
`results/b200_training_throughput_fa4_interface/`, source
`2fc1b0e4d5e472e50a641098f92b88a66f6b11da`. Its failed integration predecessors
remain in `results/b200_training_throughput_fa4/` and
`results/b200_training_throughput_fa4_dynamic/`; logs use the corresponding
`logs/runpod/` directories. FP32 master adapters remain on the network volume;
frozen contracts, metadata, selections, and logs are collected locally.

## Provisional B200 recipe and ETA

Select microbatch 1, accumulation 32, SDPA, and the 12 explicit linear
checkpoints above. Keep rank-128 FP32 master adapters, NF4/double quantization,
BF16 compute, FLA 0.5.2, selected-position logits, and the compiled surrounding
layer policy. Neither no-checkpointing, grouped batch 2, nor the tested batch-1
FA4 intervention supersedes the validated half-checkpoint gain. No combined
batching/FA4/half-checkpoint recipe is tested or recommended.

The reusable profile is
`src/gleipnir/configs/systems_screen/qwen35_4b_b200_fast.yaml`.
`experiments/b200_training_throughput/selected_config.yaml` supplies the frozen
mixed-data, matched-control reproduction contract, still limited to ten updates.
Its composed training fields, data, sample, objective, and graph bound were
audited against both completed warmed conditions. The shared cache path retains
its historical `fa4` suffix; selecting this SDPA profile does not require FA4.
The H100 profile is unchanged. Recheck the longest-row preflight for another
rank, context limit, dataset, or backend.

Ten updates of steady compute project to 185.56 seconds (3m06s), but the observed
fresh-process half-checkpoint Trainer loop took 339.03 seconds (5m39s), including
graph capture. Its complete worker took 525.54 seconds (8m46s), including loading,
saving, and adapter rebasing; the matched control worker took 507.08 seconds.
Thus the measured training-loop gain does not establish a short-job wall-time
gain across variable setup/export overhead. The warm checkpoint campaign source
is `42fd81f71343655e7a61bc655bc3a816d8b22ed3`.

One epoch over 21,837 examples at effective batch 32 is 683 optimizer updates.
Scaling the matched random-order mean of 18.556 seconds gives approximately
3.52 hours of training compute and $23.90 in GPU charges at $6.79/hour. Add
model/data setup, fresh-process capture, saving, and any difference between
this 320-row cohort and full-population lengths/order. This is a planning
projection, not a measured full epoch or a quality promotion. No full campaign
was launched.

Validation: 51 focused backend, configuration, compile, and training-command
tests passed; Ruff and launcher shell syntax checks passed. Final profile
composition was also checked against the completed conditions. Every successful
campaign's frozen manifests, metadata, summaries, and logs are collected locally;
FP32 weights and persistent kernels remain on the Pod. The agent has no verified
after-turn scheduling tool; active-turn monitoring ends with this completed screen.

Hardware and billing are described in the
[completed ID setup finding](runpod_b200_gleipnir4b_id.md): one B200 Pod at
$6.79/hour, plus $7/month network volume and $5/month running container disk.
The user requested leaving the Pod running after the screen.
