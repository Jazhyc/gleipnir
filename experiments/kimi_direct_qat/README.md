# Direct Kimi K3 quantization-aware distillation

Hypothesis: one QAT epoch directly on cached Kimi K3 distributions learns an
optimized-serving monitor without an intermediate BF16 student/teacher stage.
The user selects a fresh adapter from the pinned Qwen3.5-4B base. Reuse the
original zero-B rank-128/alpha-256 FP32 initialization, strict-filtered 7,836 rows
and 75,851,675 tokens, seed/order, logical batch 32, packing budget 16,384 and
context 29,696. Use KL T1, AdamW 5e-5, zero weight decay, 3% warmup and linear
decay for 245 updates, matching the direct-distillation baseline schedule rather
than the smaller recovery learning rate used for trained-adapter warm starts.

Use the selected original merged-weight QAT arithmetic: native NVFP4 MLP/GDN
forward, decoded quantized FP8 projection reference with FP32 contractions,
serving MXFP8 attention forward and quantized-QKV BF16 FA4 surrogate backward.
Remove unused MLP down-output recomputation and share frozen GDN QKV/Z work.
Retain BF16 FlashQLA recurrence and FP32 adapter masters/gradients. Neither
experimental native FP8 projection GEMMs nor native MXFP8 backward is enabled.

Validate the original Kimi cache checksum, teacher model, complete dataset/index
membership, source labels, strict agreement and normalized probabilities. Keep
privileged teacher prompt hashes distinct from student prompt hashes; no target
is relabeled as a same-prompt BF16 cache. No teacher calls or full teacher scoring.
Before training, score only the existing twenty training canary prompts on the
unadapted base with optimized serving, then freshly check zero-adapter simulator
alignment, packing isolation and longest-row finite-gradient/memory support.
Initial alignment has no adapter-effect requirement. Attempt 01 stops on this
initial gate; repeated unadapted serving canaries also fail. Attempt 02 preserves
that failure as a fresh-base diagnostic, reuses its checksummed server canary and
does not require initial score parity for the already validated operator recipe.
Packing isolation, longest-row and per-update finite/missing-gradient gates remain
mandatory. Final trained-adapter parity retains nonzero-effect and the original
MAE/correlation gates. Reuse checksum-bound
unchanged operator/structural receipts and persistent compiler/kernel caches.

Freeze the final one-epoch adapter only. Automatically preserve FP32 weights,
rebase/merge with FP32 accumulation into the single owned ephemeral BF16 slot,
validate new-adapter serving parity and run two frozen 3,012-row ID passes. Compare
cached optimized direct-distillation baseline, BF16 serving baseline and original
two-stage QAD (mean macro raw normalized pAUROC@20 0.87608417). The descriptive
gate permits 0.005 loss against QAD; this comparison also changes initialization,
teacher, schedule and total training budget and does not isolate teacher choice.
No ID tuning, early stopping, OOD evaluation or automatic promotion.

Stop on source/input/hardware drift, invalid targets, missing/nonfinite gradients,
failed isolation/memory/final-serving gates, OOM, unsupported execution or incomplete
coverage. Preserve failed initial alignment as diagnostic; do not call it a pass.
Cap each stage at two hours and preserve failure receipts without automatic
training retries. Launch on the existing authorized NC2 B200 with
`python -m experiments.kimi_direct_qat.launch`. Outputs are under
`results/kimi-direct-qat02/`; logs under `logs/runpod/kimi-direct-qat02/`.
The failed first attempt and its source/config archives remain under `*-qat01`.
No in-chat scheduler is available; monitor startup during the active turn, while
detached stage automation does not promise later agent wakeups.
