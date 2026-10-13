# QAT and OOD campaign shutdown

At the user's explicit request, NC2 B200 Pod `fs334v0u4zysvn` was terminated on
2026-10-13. The delete returned HTTP 204 and the subsequent Pod lookup returned
404. Network volume `ixbh81vf9c` (`gleipnir-b200-workspace`, 300 GB, US-NC-2)
was confirmed retained by a fresh read. No replacement capacity was launched.

Before termination, there were no GPU compute processes. Both completed OOD
prediction sets, parity/dispatch receipts and scoring logs were collected
locally. Seven scientific masters on the network volume matched their original
training-completion hashes: the original filtered FP4/FA4 adapter, full QAD,
QAD32, QAD64, direct-Kimi QAT, native-FP8 QAT and native-MXFP8-backward QAT.
Original FP32 adapters, inference exports, pinned bases, datasets, teacher
caches, frozen/failed contracts, results and shared compiler caches remain on
the retained volume under `/workspace/gleipnir`.

Current container runtimes, kernel sources/symlinks and remaining container-local
compiler caches were archived, with all 159,529 regular members verified against
their source files. The new archive is 9,414,040,966 bytes (8.768 GiB):
`results/b200_shutdown/20261013_qat/container_runtimes.tar.zst`, SHA256
`c4d76a280bc5f46ac63d10c3f337caea8d6b865455a49cff16ee17dc08bc4838`.
This archive remains on the network volume; its member/adapter preservation
receipts and termination response/read-back were collected locally under the
same result directory. Historical archives remain intact.

The disposable QAD merged checkpoint was excluded from preservation. Its
manifest is saved as `disposable_merge_manifest.json`; reconstruct it from the
pinned base and preserved FP32 `quantization-recovery01/merged_adapter` export.
The earlier [checkpoint cleanup](network_volume_cleanup.md) freed 60.758 GiB,
so the new runtime snapshot retains substantial net space savings.

For resumption, provision only with fresh user authorization, attach the retained
NC2 volume, restore the recorded runtimes and symlink overlays at their original
paths, and verify live hardware/runtime identities and native checks. The old
`.runpod/qat-nc2.json` is marked terminated; its SSH address is historical.
Reconstruct an adapter-specific ephemeral merge and run its parity checks before
serving. Preserve the input contracts and avoid rerunning completed controls.
Results and qualifications are recorded in [FP4/FA4 OOD](filtered_fp4_ood.md),
[QAD OOD](qad_ood.md) and the linked QAT experiment findings.
