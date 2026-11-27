# EU network volume inventory

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

The volume itself is retained; deleting it is the user's decision. The NC2
volume `ixbh81vf9c` (US-NC-2, 300 GB) is not yet inventoried: US-NC-2 had no
CPU instances and no GPU Pod at or under $0.50/hour that day.
