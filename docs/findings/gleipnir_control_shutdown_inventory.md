# `gleipnir-control` shutdown artifact inventory

This record captures the artifacts retained locally before releasing the
single-H100 `gleipnir-control` Lambda instance. The pre-termination inventory
was verified on 2026-09-01 against instance
`fd6126a47be64cae8266e1d5b3ed2b4d` (`gpu_1x_h100_sxm5`, Lambda region
`us-south-2`). The instance had no active GPU compute process or training,
evaluation, or vLLM process at the final audit. A stale read-only log follower
was still present and did not represent active experiment work.

Generated weights, predictions, datasets, and logs remain intentionally
ignored by Git. The paths, sizes, and hashes below identify their retained
local copies.

## Retained master adapters

All eight completed FP32 causal-LM master adapters were compared directly with
their remote copies. The adapter weights, `adapter_config.json`, and
`training_metadata.json` matched byte-for-byte by SHA-256. The table records
the durable master-weight identity; derived vLLM serving adapters and final-step
checkpoint duplicates remain in the collected result trees but are not the
canonical weights.

| Condition | Bytes | SHA-256 |
|---|---:|---|
| 9B, 204 matched rows | 931,170,120 | `ae54dd5cabaf6fe07fc91ec6a2c0e882be0771b1a8346c66beddaa5e29f4c6dd` |
| 9B, 504 matched rows | 931,170,120 | `16f5b58af3e885afac7ddbce066c72719552435393ccee05ca58bdb3376d415b` |
| 9B, 996 matched rows | 931,170,120 | `9470c2fbdfa526f4c706469cf9d7340ff9830da78d0b670f2ff69d7313eefef4` |
| 9B, 2,004 matched rows | 931,170,120 | `89fa5e563ad41cb70479fbbd0e74eb99e40de65b485320de90b47c4af337dfc0` |
| 9B, 4,008 matched rows | 931,170,120 | `a0b9655ed28e2397aff7180b8e90204a90b686417a9bbec7d9710d23f07cfe18` |
| 9B, 8,688 matched rows | 931,170,120 | `7fb724354abe383fd9620a5e7e410cdd79491082db8cde21a8b7166135a6160f` |
| 9B, 21,837 mixed rows | 931,170,120 | `277b27e43ac90a893a03e789232a154d1bc7f84c057eb85ec9cc4e5eb60d7cdf` |
| 4B, 21,837 mixed rows | 679,511,752 | `5c4688931ac9168e0d27aa501b73141180b74fe01e0ec147448f7661f16c7bc6` |

The mixed 4B and 9B masters are additionally staged locally under
`results/huggingface_release_mit/` and published as the MIT-licensed Gleipnir
4B and Gleipnir 9B releases.

## Retained result and log trees

The path-and-size manifest hash is SHA-256 over each tree's sorted relative
paths and decimal byte sizes. It is an inventory identity rather than a
content hash; the canonical master weights above were content-hashed in full.
The remote and local file counts and total sizes matched before termination for
the scaling, mixed-4B, and distilled-OOD result trees.

| Local tree | Files | Bytes | Path/size manifest SHA-256 |
|---|---:|---:|---|
| `results/tool_trajectory_distillation_scaling/` | 259 | 31,090,782,881 | `fa0e5a7ec85dfffee00e1f32086630742a67d15482b92ba11fc0c6b15fd9ccfc` |
| `results/tool_trajectory_distillation_mixed_qwen4b/` | 53 | 4,157,255,185 | `13564446052ebb87dfcd48848428ce4955bcbf9577b4c5944c599d3868cd16b0` |
| `results/tool_trajectory_distillation_ood/` | 37 | 28,791,503 | `5ef7c179d67c957ecfaafa875a6bd13bc7e4a167ee4e8636742eab1a98b17b27` |
| `results/tool_trajectory_distillation_full_teacher_prompt_ood/` | 6 | 3,664,231 | `8307a0558288559c286c10340c294d654374bfd1944d8bed1f89d945c288d66e` |
| `logs/tool_trajectory_distillation_scaling/` | 80 | 281,898 | `9681b2379c31e2164c5319da733645981bcfce523787795303575394b9dfdac3` |

The last collection pulled four previously missing full-prompt canary result
and prediction files totaling 58,169 bytes, plus all 80 Hydra training logs and
resolved configurations totaling 281,898 bytes. The existing campaign logs,
job records, manifests, parity reports, complete OOD predictions, and summaries
were already local.

## Deliberately not retained

The remote `results/tool_trajectory_distillation_mixed_qwen08b/` tree occupied
2,389,656,612 bytes and consisted primarily of duplicated one-step systems
preflight adapters. The planned 0.8B campaign was intentionally stopped at step
12 of 683, produced no completed training checkpoint, and ran no OOD evaluation.
Its durable stop record, launcher status, manifest, job records, and aggregate
campaign logs are local. The preflight weights are not model candidates and
were not retained, consistent with
`findings/tool_trajectory_qwen08b_stopped_systems_preflight.md`.

Model caches, the isolated FLA installation under `/tmp`, the synced source
snapshot, and reconstructable input materializations were also not duplicated.
They are reproducible from pinned revisions, committed code, recorded prompt
hashes, and the retained manifests.

## Termination confirmation

Termination was explicitly authorized by the user on 2026-09-01. The Lambda
API accepted the request for instance `fd6126a47be64cae8266e1d5b3ed2b4d`
and reported it as `terminating`. Repeated follow-up queries subsequently
returned no matching `gleipnir-control` instance, confirming that the H100 was
released.
