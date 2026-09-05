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

The 122B run started successfully on 2026-09-06. The exact ID audit contains
37,077,317 prompt tokens (maximum 30,325). FP8/TP2 was verified at runtime;
weights used 57.83 GiB per GPU and the memory profiler retained 10.72 GiB per
GPU for KV cache. Cold initialization took 642.46 seconds, including compiler
and DeepGEMM warmup. The five-row balanced-plus-longest canary passed:
batch/singleton mean absolute probability difference 0.007415, maximum
0.026606. Full evaluation then started in the same engine. vLLM reported a
default, rather than shape-tuned, MoE kernel configuration, limiting claims
about best achievable throughput. These are startup findings, not final scores.

## Authorized sequential follow-up: Qwen3.8-27B FP8

On 2026-09-06 the user added Qwen/Qwen3.8-27B-FP8 after the current 122B
evaluation. Revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a has
30,866,866,928 weight-file bytes. The installed vLLM 0.24.0 supports its
Qwen3_5ForConditionalGeneration architecture and explicit swish GDN output gate.
Its tokenizer/chat boundary and full ID token lengths are audited independently.

The additional Hydra config inherits the same FP8/TP2, full-rubric,
non-thinking, Triton, no-prefix-cache, batch/concurrency, metrics and canary
contract. Neither this model's quality nor its speed is assumed to exceed the
122B model. Its smaller weights should free more memory for KV cache, but this
fixed-concurrency screen is not a maximum-throughput or prefix-cache benchmark.
Compare both completed ID endpoints; no new annotation, training or OOD work.

Separate artifacts/logs are under `results/qwen38_27b_id/` and
`logs/lambda/qwen38_27b_id/`. The dependency gate checks predecessor success and
complete coverage, then waits for GPU release. It fails on predecessor failure
and never preempts the current run. Queued checks run every ten minutes.

```bash
python -m experiments.qwen122b_id.run prepare --config-name qwen38_27b
python -m experiments.qwen122b_id.run run \
  --result-dir results/qwen38_27b_id --after-result-dir results/qwen122b_id
```
