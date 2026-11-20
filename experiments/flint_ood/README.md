# Flint on the strict OOD suite

Could [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint) succeed
Qwen3.8-27B as the base for the large monitor? Flint is a 39.8B
mixture-of-experts model with about 2.5B parameters active per token
(Apache 2.0, released 2026-10-21, revision `2a247be0…`). This experiment scores
the unadapted checkpoint under the contract that chose Qwen3.8-27B
([Qwen 27B generations](../qwen27b_generations_ood/README.md)) and measures its
scoring throughput beside Qwen3.8-27B on the same GPU.

## Pre-registration (frozen 2026-11-20, before any scoring)

- **Hypothesis.** Unadapted Flint reaches Mean-OOD pAUROC@20 within 0.02 of
  unadapted Qwen3.8-27B (0.8901) at the direct-logit interface, at higher
  scoring throughput on the same B200.
- **Intervention.** The checkpoint changes to Flint BF16, and with it what the
  checkpoint forces: its chat template and tokenizer, and vLLM 0.34.0 (Flint
  needs 0.33 or later; the project runtime is 0.31). Unchanged from the Qwen
  27B contract: the 6,395 rows, the full teacher prompt, the `Prediction:`
  prefix, literal `0`/`1` logprob scoring, and the engine settings (32,768
  maximum length, 16 sequences, 32,768-token batches, prefix caching, seed
  20260830).
- **Decision boundary.** Flint has no thinking switch; it chooses per turn
  whether to open with `<think>`. The prompt is one user turn followed by
  `<|assistant|>Prediction:`, so the scored turn carries no reasoning. This is
  the interface a student would be trained at. The canary records, as a
  diagnostic only, the unconstrained first assistant token.
- **Baselines.** Qwen3.8-27B 0.8901, Qwen3.6-27B 0.8726 and Qwen3.5-27B 0.8388
  (2026-10-17, vLLM 0.31).
- **Same-stack control.** Qwen3.8-27B BF16 is rescored on the same Pod with
  vLLM 0.34.0 under its frozen contract (`qwen38_27b_vllm034.json`). It is the
  throughput reference, and its scores are compared with the stored vLLM 0.31
  predictions (score MAE, correlation, Mean-OOD difference) to show whether the
  stack change moves scores.
- **Reporting.** Mean-OOD pAUROC@20 (unweighted mean of six sources),
  per-source pAUROC@20 and AUROC, macro Brier, balanced accuracy, score ties;
  scoring seconds, rows/s and prompt tokens/s.
- **Reading rule.** Descriptive. "Competitive" means Mean-OOD pAUROC@20 of at
  least 0.8701. The user decides any follow-up (ID scoring, adapter training).
  Nothing is tuned on these labels and the result is reported whatever it is.
- **Stop conditions.** Input, manifest, revision, teacher-prompt, rendering or
  boundary drift; truncation or a token-count mismatch; a missing decision
  logprob; a 13-row canary (one row per source and label, plus the longest
  prompt) that is incomplete, non-finite, or whose batched and singleton
  scores differ by more than 0.02 on average or 0.05 at most; missing or
  duplicate rows.
- **Compute.** User authorization 2026-11-20: one B200 Pod, EU-RO-1 preferred
  so that throughput is comparable with the earlier Qwen3.8-27B runs.

## Interface audit (`audit.py`, CPU only)

| | Flint | Qwen 27B |
|---|---:|---:|
| prompt tokens, total | 50,935,936 | 60,539,619 |
| prompt tokens, maximum | 26,047 | 30,382 |
| decision tokens `0` / `1` | 15 / 16 | 15 / 16 |

- Flint's tokenizer needs 16% fewer tokens for the same prompts.
- The boundary tokenizes as `<|assistant|>`, `Prediction`, `:`, then the digit.
  The template uses no BOS token, and the tokenizer adds none.
- The template sanitizes user text. It escapes role markers, which rewrites all
  1,499 Rogue Eval prompts (their transcripts contain literal `<|user|>` and
  `<|assistant|>`), and it removes NUL characters from 34 CUA-SHADE-Arena
  prompts. No prompt contains think tags. This is the checkpoint's own
  interface and is kept.
- Rendered-prompt SHA-256: `b2786090…` (frozen in `flint.json`).

## Why not the hosted endpoint

OpenRouter's `recursive/flint-v1.0:free` cannot give this interface (probed
2026-11-20 with synthetic prompts only): reasoning is mandatory, an assistant
prefill is treated as an earlier turn, the completions route is wrapped into
chat, and the account is limited to 1,000 free requests per day.

## Commands

```bash
PYTHONPATH=src:. python -m experiments.flint_ood.audit   # tokenizer only
# on the Pod, after syncing the code and the OOD input
experiments/flint_ood/setup.sh flint qwen38
CANARY_ONLY=1 experiments/flint_ood/run.sh flint
experiments/flint_ood/run.sh flint
experiments/flint_ood/run.sh qwen38
```

## Results

Pending.
