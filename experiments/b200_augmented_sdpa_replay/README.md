# Historical augmented adapter replay with BF16/SDPA

User-authorized seed-0 replay of the preserved historical augmented adapter.
Hypothesis: restoring historical segmented causal SDPA full attention reproduces
its checkpoint and explains the current BF16/FA4 training gap. The current
seed-0 BF16/FA4 repeat was bitwise identical, so its observed execution noise
does not explain that gap. This test holds initialization, rows/targets, order,
batching, optimizer and one-epoch selection fixed; it does not vary data seeds.

Use the preserved `qwen35_4b_b200_packed_sdpa` profile, BF16 MLPs and FlashQLA,
original zero-B rank-128/alpha-256 FP32 initializer, 8,688 augmented rows,
83,969,213 training tokens, logical batch 32/272 updates, 16,384 packing budget,
29,696 context cap, AdamW 5e-5, linear decay/3% warmup and no dropout/weight
decay/checkpointing. Preserve selective compilation, BF16 reduction controls,
runtime and persistent caches. Reuse the original SDPA startup receipt. SDPA
may internally select a FlashAttention kernel; the intervention removes the
explicit packed FA4 implementation, not every possible flash kernel.

Before launch, compare the resolved job against the frozen historical job;
only output/run bookkeeping and added initializer/receipt checksum assertions may
differ. After training, verify all 4,556 physical batches, data order, token
counts, SDPA metadata, optimizer and FlashQLA kernel source hashes. Compare
both FP32 master and serving-export hashes and semantically equivalent configs
(unordered target-module serialization may differ). Preserve all artifacts.

If both adapters exactly match historical weights/configs, reuse the completed
same-host current-stack BF16 ID scores and skip merge/reference/evaluation.
Otherwise preserve the changed checkpoint, merge in FP32 to a separate ephemeral
BF16 checkpoint and require fresh master/merged/serving parity and native BF16
audits before one matched ID pass. Limits remain MAE <=0.020/correlation >=0.99,
finite scores and nonzero adapter effect; no diagnostic waiver transfers.
Keep native tokenizer, cached causal LAST 0/1 head, corrected scheduler, prefix
off, c128 and original 128-row groups over 3,012 CoT-removed ID prompts and
33,750,959 input tokens. Compare to historical SDPA and current BF16/FA4 BF16
controls; report macro/pooled/source AUROC/raw pAUROC20, calibration/thresholds,
paired shifts and input tokens/s/latency if fresh scoring is necessary.

Select the final complete checkpoint only. Stop on recipe/identity/runtime drift,
nonfinite/missing gradients or scores, OOM, failed gates/native dispatch,
coverage failure or completion. Do not select or promote on ID. Exact historical
identity would support the attention implementation as the training difference;
failure to reproduce leaves other historical/current source/runtime/host
differences unresolved. No APPS, JudgeDeceiver, capacity change or new seed.
Use the allocated idle NC2 B200. Merged weights stay on its ephemeral storage
or persistent-volume archive; collect adapters, outputs and receipts locally.
No heartbeat tool is available; monitoring is active-turn only.

Run `python -m experiments.b200_augmented_sdpa_replay.run --stage prepare`, then
`--stage run` from the preserved training Python with `PYTHONPATH=src:.`.
The campaign runner launches each worker in its pinned runtime. Artifacts use
`b200-augmented-sdpa-replay01`; a new merge is made only if needed, at
`/tmp/gleipnir-merged/historical-augmented-sdpa-replay01`.
