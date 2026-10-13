# Network-volume checkpoint cleanup

On 2026-10-13 the user authorized removing unnecessary merged checkpoints
while OOD evaluation runs. `scripts/prune_reconstructable_merges.py` verifies
the pinned base-file hashes, adapter/config hashes and FP32 adapter tensors
before deleting archived merged model shards from three fixed archive roots.
The active ephemeral evaluation model is outside the cleanup scope.

Seven checkpoints were pruned successfully, removing 60.758 GiB; remote
`results/` usage fell from 169 GiB to 108 GiB. They are
`caft-regular-sdpa02`, `activation-filter-ranked20-sdpa01`,
`historical-augmented-sdpa-replay02`, `fp4-full-training-bf16`,
`historical-augmented-bf16`, `bf16-augmented-fa4` and `fp4-augmented-bf16`.
All verified bases, FP32 adapters, configs, tokenizers, manifests, datasets,
compiler/kernel caches and result receipts remain. Each directory retains a
`weights_pruned.json` reconstruction receipt. The plan, incremental deletion
receipt and completion receipt are under `results/network-volume-cleanup01/`.
Runtime archives and the compressed concept-ablation merge were retained.

Four focused tests pass, covering metadata preservation and rejection of
adapter/base drift or reduced-precision adapters; Ruff passes. No capacity or
volume lifecycle operation was performed.
