# Latest-only weights for short systems training checks

Decision date: 2026-10-06. After the network-volume audit, the user explicitly
requests overwriting the same adapter for systems training optimization and
authorizes deleting expendable prior short-screen weights. Save per-trial
configs, losses, timing, gradient checks, tensor digests and numerical receipts.
Keep one frozen original initialization, all compiler/kernel caches, and full
research/quality/evaluation adapters. This policy does not discard meaningful
trained models or authorize capacity termination.

The exact remote deletion plan removes 127 named adapter/master files from
short BF16/FA4/FP4/NF4/GDN systems screens, totalling 86,300,534,628 apparent
bytes. Every planned file is verified absent and all 48 protected weight files
present. The final FP4 full-training/serving adapter and its initialization stay
intact. Results metadata, logs and caches remain. Retire matching collected
local copies too: 81 files/links, 51,645,138,872 apparent content bytes, with
all 44 locally present protected files retained. Preserve the plans and receipts
under `results/b200_inference_benchmark/`; no new historical parity pass is claimed.

`src/gleipnir/systems_artifacts.py` writes diagnostic masters atomically to the
shared `results/systems_training_scratch/fp32_master.pt`, retaining the previous
file on failure. Precision screens and execution audits record a mutable
latest-only pointer while keeping their per-trial digests. Resident Trainer
trials overwrite `results/systems_training_scratch/adapter/` rather than creating
an adapter for every twenty-step trial.

Ordinary Trainer screens opt in with `systems_adapter_scratch: true` in the
screen configuration/job. The launcher forwards it to the trainer, which
suppresses checkpoint copies, redirects weights/tokenizer to the shared slot
and keeps metadata in the individual run directory with an explicit mutable
artifact reference. Historical frozen screen configs and full quality jobs
leave this flag absent, preserving their execution contracts. Future systems
work must opt in, as recorded in `AGENTS.md`.

No new GPU training run is launched for this storage-only change. Focused CPU
checks cover atomic replacement/failure preservation, explicit launcher routing,
historical-job preservation, and existing optimizer/reset numerical behavior.
