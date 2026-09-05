# Qwen3.5-122B-A10B FP8 ID teacher screen

Frozen question (2026-09-06): can the official FP8 MoE provide useful local
teacher scores at practical throughput? Evaluate all 3,012 existing ID rows
(946 STRIDE test, 2,066 Gloom exfiltration). No OOD reads, annotation campaign,
training, or prompt/threshold tuning is authorized by this screen.

Use the full existing teacher rubric, non-thinking Qwen chat wrapper and
selected literal 0/1 logprobs, generating one token. The official FP8 revision
is a099dee70ccfcd8d5dda56aaa0b60cb8ecadabc9 (127,163,011,832 weight-file bytes).
Both reserved Lambda GPUs are H100 80GB HBM3, driver 580.126.20. TP2 is required;
skip the vision tower, retain BF16 activations and checkpoint-declared dynamic
block FP8 quantization. No BF16-versus-FP8 quality equivalence is claimed.

Initial systems contract: 32K context, eight concurrent sequences, 8,192
batched prefill tokens, GPU-memory utilization 0.90, Triton GDN prefill and
prefix caching disabled. This new model has no historical backend contract;
Triton avoids the recurring FlashInfer GDN evaluation stalls. This is not a
prefix-cache throughput test or a matched speed comparison against 27B.

Before full scoring, require the four source-label canary rows plus the longest
ID request to return finite decision scores and fit without truncation. Repeat
those requests individually; require batch/singleton mean absolute probability
error <=0.02 and maximum <=0.05. This checks numerical consistency, not accuracy
or FP8 quantization parity. Keep the original batch scores, record repeats
separately, and evaluate the remainder in the same persistent engine.

Report source-macro and per-source pAUROC@20, AUROC, Brier, threshold diagnostics
and ties. Existing ID teacher/student scores are contextual baselines; models
with different prompts or precision are not matched capacity ablations. The
old 27B OOD score is not an ID comparator. Stop after the fixed full endpoint;
no automatic teacher selection or billion-token annotation. Stop on provenance
drift, tokenizer audit drift, GPU contention, OOM, missing logits, failed
canary, nonfinite scores, or 20 minutes without prediction progress after the
canary. Initialization has a separate 30-minute no-progress allowance.

Hydra authoring resolves once to a hashed JSON execution contract. The thin
entrypoint reuses the existing Qwen evaluator, preserves resume identities,
records GPU-memory samples, and owns a ten-minute progress-checking heartbeat
with a bounded stall failure (not merely a log timestamp).

```bash
python -m experiments.qwen122b_id.run prepare
python -m experiments.qwen122b_id.run run
```

Artifacts: `results/qwen122b_id/`; logs: `logs/lambda/qwen122b_id/`.
