# NVIDIA MXFP8 full-attention training screen

Hypothesis: NVIDIA's Blackwell D256 MXFP8 causal GQA forward/backward engines
reduce complete Qwen3.5-4B LoRA update time relative to the user-selected BF16
FA4 recipe. Test eight full-attention layers; retain 24 BF16 FlashQLA GDN layers,
BF16 MLPs, FP32 rank-128 master adapters and the frozen systems-screen cohort.

The initial intervention calls dense BSHD MXFP8 separately for each example
within the existing packed physical row. It preserves sequence boundaries and
logical example weighting but sacrifices fused variable-length attention. All
conversion, scale-layout repacking, allocation and backward costs are included.
Pin cuDNN Frontend revision `51d9d06b574222378a3d806009accab098e73705` in an
isolated overlay; preserve the current environment and shared persistent caches.

Before model loading, check actual causal D256 GQA 16Q/4KV forward/backward
against independent FP32 attention, quantizer layout against NVIDIA's reference,
and cross-example isolation. Record within-sequence future-token quantization
effects separately. Forward relative L2 limit is 2%, gradient relative L2 5%.
Stop on unsupported engines, missing/nonfinite gradients, isolation failures,
native numerical gate failure, OOM or dependency incompatibility; preserve
failed receipts. No silent fallback, tolerance tuning or default promotion.

If native checks pass, perform fresh eager/compiled model packing checks and
longest-cohort memory preflight, then compare matched 20-update trajectories
from the same initial adapter, ten warmup and ten measured updates. Select no
checkpoint on held-out quality; this is a bounded systems screen, with no teacher
requests. Recommend further work only if complete measured updates are at least
5% faster; longer quality checks and replication remain necessary for promotion.

Artifacts: `results/b200_nvidia_mxfp8/`. Logs:
`logs/runpod/b200_nvidia_mxfp8/`. Startup and bounded execution are monitored in
the active turn; this session has no in-chat scheduling tool for later follow-ups.

```bash
bash experiments/b200_nvidia_mxfp8/bootstrap.sh
```

The bootstrap builds the pinned upstream snapshot from an ignored source archive
and resolves cuDNN 9.26 separately from the existing runtime. Exact installed
versions, archive checksum and effective cache paths must be recorded before
kernel execution. It does not modify `pyproject.toml` or `uv.lock`.
