# Teacher filtering campaign shutdown

At the user's request, Runpod B200 `rzarg8j8x0so3i`
(`gleipnir-teacher-delta-sweep`, US-NC-2) was terminated on 2026-10-12.
The delete returned HTTP 204; the subsequent Pod lookup returned 404.
Network volume `ixbh81vf9c` (`gleipnir-b200-workspace`, 300 GB, US-NC-2)
was confirmed retained. No further experiments were launched.

Before termination, the GPU had no compute processes. Seven scientific adapter
sets passed master/export SHA256 verification on the mounted network volume:
the four delta arms, the partial-AUROC candidate, and the original regular
4B/9B checkpoints used for the final teacher-fit comparison. Results, raw
predictions, frozen contracts, failed receipts, logs and post-hoc audits were
collected locally; original weights, datasets and persistent caches remain
under `/workspace/gleipnir` on the retained volume.

Preservation and termination receipts are in
`results/b200_shutdown/20261012_teacher_sweep/`. The new container source/cache
archive contains 123 verified regular files and was collected locally; SHA256:
`a6267fb325050cc905a38ff4eb1e559d5a210d73a7c5233a212ae9f886b57543`.
The unchanged pinned runtime archive remains on the volume at
`results/b200_shutdown/20261009_nc2/container_runtimes.tar.zst`; its verified
SHA256 is `7ac16b76adab107fdd435949deac24b93afb94239138ef0ba87aa0a6b2f51366`.
The disposable final 9B merge was not retained; its manifest is saved and it can
be reconstructed from the persistent base and original adapter.

Scientific outcomes and qualifications remain in the
[partial-AUROC findings](partial_auroc.md) and
[training-loss comparison](monitor_training_loss_fit.md), with their linked
experiment records. Historical launch/PID notes are not live capacity inventory.
