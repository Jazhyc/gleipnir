# Runpod network volume inventory

Status: completed.
Date: 2026-11-27.

User request, 2026-11-27: inventory Runpod network volume `cb7bxu71ug`
(EU-RO-1, 200 GB; unused since 2026-11-04, with 55 unreadable files since
2026-11-03) against the Storage Box before deciding whether to keep it.

**Method.** A CPU Pod (`hjk0msbi8ytw7e`, cpu3c-4-8, $0.12/hour) listed every
file (158.3 GB in 359,180 files) and hashed every model-like file
(379 files, 99.6 GB: `.safetensors`, `.pt`, `.bin`, `.npy`, archives). The
Storage Box tree (`research/gleipnir/`, from `rsync --list-only`) was matched by
path and size, and every candidate box copy of a model file was hashed on the
box. Records: `logs/runpod/volume_inventory_20261127/` (manifest, both hash
sets, verdicts, `compare.py`).

**Result.**

| Group | Files | GB |
|---|---:|---:|
| Model files byte-identical on the box | 152 | 59.9 |
| Other files on the box at the same path and size | about 25,500 | about 63 |
| Rebuildable: virtualenvs, compiler/kernel caches, HF downloads (includes the unreadable files) | 326,951 | 41.3 |
| Only on the volume | about 830 | about 37 |

Only on the volume were the FP32 master and serving adapters of four completed
4B runs: `gleipnir27b-teacher-teacherprompt-s1-01`,
`gleipnir27b-teacher-student-s1-01`, `filtered-fp4-fa4-baseline01` and
`teacher-disagreement-delta000-sdpa04` (16 files, 5.4 GB with their
`training_args.bin`). At the user's selection ("masters only") they were copied
to their recorded paths under `research/gleipnir/results/`, adding files only,
and verified on the box by SHA-256 against the volume (16 of 16 match;
`masters_copied.sha256`).

Not copied (user selection): `probe_optimized_serving` activation dumps
(13.2 GB), runtime archives (14.7 GB: QAT container runtimes, the 2026-10-07 EU
serving runtime, container-cache archives), the 27B-as-teacher and Qwen3.8
teacher training datasets and the probe prompt sets (2.5 GB),
`quantization-recovery01` merged and serving copies (1.0 GB; its master is on
the box), `systems_training_scratch/fp32_master.pt` (2.6 GB scratch; the box
copy is a header-only stub) and 27B vocab-distillation hidden states (63 MB).

At the user's instruction the EU volume was then deleted (2026-11-27 05:42
UTC, `deleteNetworkVolume`; no Pods attached, absent from
`myself.networkVolumes` afterwards).

## NC2 volume `ixbh81vf9c`

Same method, on CPU Pod `m6j0cnb198pfck` (cpu3c-4-8, $0.12/hour, available on
the second attempt that morning): 203.3 GB in 452,723 files, of which 106.1 GB
in 657 model files (over 1 MB) hashed on both sides.

| Group | Files | GB |
|---|---:|---:|
| Model files byte-identical on the box | 522 | 54.2 |
| Model files only on the volume | 135 | 51.9 |
| Rebuildable (environments, caches, HF downloads) | 382,633 | 72.6 |

Only on the volume were the FP32 master and serving adapters of six completed
4B runs: `activation-filter-matched20-sdpa01`,
`activation-filter-ranked20-sdpa01`, `strict-pauc02` and
`teacher-disagreement-delta{005,010,020}-sdpa04` (24 files with their
`training_args.bin`). Under the same "masters only" selection they were copied
to their recorded paths and verified on the box (24 of 24 match;
`nc2_masters_copied.sha256`). The NC2 copies of `filtered-fp4-fa4-baseline01`
and `teacher-disagreement-delta000-sdpa04` are byte-identical to the EU copies
already preserved, and every `checkpoint-*` adapter equals its run's final
master.

Not copied: runtime archives (`b200_shutdown/20261009_nc2`, `20261013_qat`;
20 GB), the concept-ablation merged-checkpoint archive (7.3 GB, derivable from
its master), CAFT direction-refit activation caches (1.5 GB), merged and
serving copies of the six QAT/QAD runs whose masters are on the box (6.1 GB),
the scratch `systems_training_scratch` adapter, and the external
`monitor-injection-evals` checkout's small result files (under 0.1 GB). The NC2
volume is retained; deleting it is the user's decision.
