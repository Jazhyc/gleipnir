# Flint on the strict OOD suite

Could [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint) succeed
Qwen3.8-27B as the base for the large monitor? Flint is a 39.8B
mixture-of-experts model with about 2.5B parameters active per token
(Apache 2.0, released 2026-10-21, revision `2a247be0…`). This experiment scores
the unadapted checkpoint under the contract that chose Qwen3.8-27B
([Qwen 27B generations](../qwen27b_generations_ood/README.md)) and measures its
scoring throughput beside Qwen3.8-27B's on the same GPU type.

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
- **Throughput reference.** The stored Qwen3.8-27B run: 2,190 s per scoring
  invocation on an EU-RO-1 B200 with vLLM 0.31. The comparison therefore
  includes the vLLM version change (0.31 to 0.34) and a different physical
  host of the same type.
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

### Amendment (2026-11-20, during the canary, before any full-suite score)

The first version planned a same-stack control: Qwen3.8-27B rescored on this
Pod with vLLM 0.34.0. The user dropped it; it was never run. Throughput is
compared with the stored vLLM 0.31 result instead. `flint.json` was frozen
with the control listed as its third baseline and is left unchanged so that
the pre-registration stays as written.

A first canary attempt failed in engine warm-up before any row was scored:
vLLM 0.34.0 defaults to 64-token KV-cache blocks, and Flint's decode kernel
asserts on anything but 16. `flint.json` now sets `block_size: 16`; this
changed its hash before any prediction existed.

The second canary attempt scored its 13 rows and then stopped on the
batched-versus-singleton gate (mean 0.039, maximum 0.153 in score).
`diagnose_parity.py` located the cause without reading labels. With prefix
caching off, a batch, singletons and a reversed batch give bit-identical
margins. With it on, the same prompt moves by up to 0.74 nats between passes
(mean 0.08 to 0.23), including between two identical batches. Prefix-cache
hits therefore change Flint's logits in vLLM 0.34.0. `flint.json` now sets
`enable_prefix_caching: false`, the one engine setting that departs from the
Qwen contract; the failed canary is kept as
`results/flint_ood/flint_canary_prefix_cache_failed/`. Qwen3.8-27B's stored
throughput had prefix caching on, so the throughput comparison carries that
difference as well.

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
PYTHONPATH=src:. python -m experiments.evaluation.flint_ood.audit   # tokenizer only
# on the Pod, after syncing the code and the OOD input
experiments/evaluation/flint_ood/setup.sh
CANARY_ONLY=1 experiments/evaluation/flint_ood/run.sh flint
experiments/evaluation/flint_ood/run.sh flint
# consistency diagnostic used for the prefix-caching amendment
python -m experiments.evaluation.flint_ood.diagnose_parity --output results/flint_ood/diagnostics/parity_no_prefix_cache.json
```

## Results (2026-11-20, EU-RO-1 B200 `GPU-a2b24934…`, vLLM 0.34.0)

Pod `8l2ln36loy87kt` (EPYC 9555, driver 595.112.07). The run scored all 6,395
rows and hit no stop condition. The canary's batch and singleton scores were
identical. Engine start-up took 41 s with warm kernel caches; the first start
spent about 9 minutes on one-off kernel builds and autotuning.

**Flint is not competitive at the direct-logit interface.** It is 44 points
below Qwen3.8-27B and 17 below unadapted Qwen3.5-4B.

| strict OOD | Flint | Qwen3.8-27B | Qwen3.5-4B, unadapted |
|---|---:|---:|---:|
| **Mean-OOD pAUROC@20** | 0.4486 | **0.8901** | 0.6175 |
| macro AUROC | 0.7819 | **0.9535** | 0.8549 |
| macro Brier | 0.2268 | **0.0823** | |
| macro balanced accuracy | 0.6361 | **0.8894** | |

| pAUROC@20 by source | Flint | Qwen3.8-27B |
|---|---:|---:|
| AgentDojo | 0.4539 | 0.9657 |
| CUA-SHADE-Arena | 0.5244 | 0.8648 |
| IaC | 0.1625 | 0.9678 |
| MLE-Sabotage | 0.6735 | 1.0000 |
| Rogue Eval | 0.3951 | 0.9862 |
| SHADE-Arena | 0.4824 | 0.5560 |

The Qwen3.5-4B figure is the archived zero-shot teacher-prompt result
([finding](../../../docs/findings/evaluation/tool_trajectory_qwen4b_ood_benchmark.md)).

- **Flint over-flags.** At 0.5 it flags 98% of benign Rogue Eval rows, 92% of
  SHADE-Arena and 69% of AgentDojo. Mean scores on benign rows are 0.30 to
  0.70 by source, and Flint's scores correlate with Qwen3.8-27B's at only 0.50.
- **The direct turn is off-distribution for Flint.** On all 13 canary prompts
  the unconstrained first assistant token is `<think>` with probability above
  0.99999. Flint practically never answers this prompt without reasoning, so
  this measures the unadapted model at a boundary it does not use. It says
  nothing about Flint with reasoning or after adapter training.

**Throughput.**

| scoring the 6,395 rows | Flint | Qwen3.8-27B |
|---|---:|---:|
| seconds | 364 | 2,190 |
| rows/s | 17.6 | 2.9 |
| prompt tokens/s (own tokenizer) | 139,800 | 27,600 |

- Flint is 6.0 times faster per row. The comparison is not matched: Flint ran
  on vLLM 0.34.0 without prefix caching, Qwen3.8-27B on vLLM 0.31 with it, on
  different EU-RO-1 B200 hosts.
- Against the 4B, see the agent-trajectory benchmark below.
- **Tokenization is not the bottleneck** (`throughput.py`). Scoring from
  pre-tokenized IDs took 363 s, the same as from text, with identical scores.
  The stock tokenizer needs 73 s on one thread for the 50.9M tokens (700K
  tokens/s), which vLLM overlaps with GPU work. Gigatoken 0.10.0 and 0.11.0
  cannot load Flint's tokenizer (unsupported Split pre-tokenizer regexes), so
  it was not run; the token-ID pass is the ceiling it could reach.

**Limits.** One unadapted checkpoint, one prompt interface, one pass. ID was
not scored. Prefix caching changes Flint's logits in vLLM 0.34.0 (see the
amendment), which would need resolving before serving it with a shared prefix.

Artifacts: `results/flint_ood/` (`flint/`, `comparison.json`,
`diagnostics/`, the failed canary) and `logs/runpod/flint_ood/` (25 files
verified against the first Pod by SHA-256, `collect.sha256`; 35 against the
second, `collect_pod2.sha256`). Pod lifecycle:
[infrastructure](../../../docs/infrastructure/runpod.md#runpod).

## Second arm: empty think block (frozen 2026-11-20 before scoring)

At the user's suggestion: close an empty reasoning block before the decision
prefix, as the Qwen contract does. The prompt ends
`<|assistant|><think></think>Prediction:`, using the form in which Flint's
template writes reasoning before an answer (no newlines).

- **Hypothesis.** The empty block moves Flint out of the reasoning-opening
  state and lifts Mean-OOD pAUROC@20 above the bare boundary's 0.4486.
  "Competitive" is still 0.8701 or more.
- **Intervention.** Only the prefill changes (`flint_empty_think.json`); the
  rows, prompt, scoring and engine settings are those of the first arm. The
  canary also records the unconstrained first token after `</think>`.
- **Audit.** 50,948,726 prompt tokens (two more per row), maximum 26,049,
  decision tokens 15 / 16, rendered SHA-256 `7630c888…`.
- **Caveat.** The arm was added after the first arm's OOD score was known. It
  is one fixed prefill; no other prefills are tried on these labels, and both
  arms are reported.

```bash
experiments/evaluation/flint_ood/run.sh flint_empty_think
```

### Result (2026-11-20, same B200 `GPU-a2b24934…`, Pod `j59qldo1255y10`)

All 6,395 rows scored, no stop condition, batch and singleton canary scores
identical, 363 s of scoring.

| strict OOD | bare boundary | empty think | Qwen3.8-27B | Qwen3.5-4B, unadapted |
|---|---:|---:|---:|---:|
| **Mean-OOD pAUROC@20** | 0.4486 | 0.4811 | **0.8901** | 0.6175 |
| macro AUROC | 0.7819 | 0.8025 | **0.9535** | 0.8549 |
| macro Brier | 0.2268 | 0.2356 | **0.0823** | |
| AgentDojo | 0.4539 | 0.4897 | 0.9657 | |
| CUA-SHADE-Arena | 0.5244 | 0.5140 | 0.8648 | |
| IaC | 0.1625 | 0.1793 | 0.9678 | |
| MLE-Sabotage | 0.6735 | 0.7806 | 1.0000 | |
| Rogue Eval | 0.3951 | 0.4657 | 0.9862 | |
| SHADE-Arena | 0.4824 | 0.4572 | 0.5560 | |

- **The empty block helps a little: +3.2 points**, with gains on four sources
  and small losses on CUA-SHADE-Arena and SHADE-Arena. Flint stays 41 points
  below Qwen3.8-27B and 14 below unadapted Qwen3.5-4B. Not competitive.
- **The boundary is now one Flint uses.** After `</think>` the unconstrained
  next token is `Prediction` with probability 0.995 or more on all 13 canary
  prompts (the runner-up, `We`, is at most 0.4%). So the weak result is not
  explained by an off-distribution boundary: the unadapted model does not
  rank these trajectories well without reasoning.
- The two arms' scores correlate at 0.84. Both over-flag (mean score 0.67
  to 0.69; 90% of benign SHADE-Arena rows are flagged at 0.5).

## Serving throughput on the 96 agent trajectories (`agent_bench.py`)

User request: a direct throughput measurement on the report's serving
benchmark ([serving concurrency](../../serving/serving_concurrency/README.md), `--agent`:
BashArena, BashBench, Gloom and STRIDE with the compact instruction, closed
loop, three passes per level after one warmup pass, prefix caching off, 32,768
maximum length and batch budget, 128 sequences). Flint is served by stock
`vllm serve` 0.34.0 in BF16 through `/v1/completions` (one constrained token
with logprobs, empty-think boundary). Flint tokenizes these rows to 1,064,479
tokens (median 9,040) against Qwen's 1,193,325, so requests/s is the
like-for-like unit.

| one request at a time | Flint BF16 | Qwen3.5-4B BF16 | Qwen3.8-27B BF16 |
|---|---:|---:|---:|
| requests/s | 9.6 | 8.3 | 1.7 |
| prompt tokens/s (own tokenizer) | 106,319 | 103,266 | 21,720 |
| latency p50 / p95, ms | 82 / 226 | 94 / 267 | 452 / 1,281 |

| 8 concurrent requests | Flint BF16 | Qwen3.5-4B BF16 | Qwen3.8-27B BF16 |
|---|---:|---:|---:|
| requests/s | 12.3 | 9.7 | 1.8 |
| prompt tokens/s (own tokenizer) | 136,701 | 120,284 | 22,950 |
| latency p50 / p95, ms | 627 / 907 | 816 / 1,131 | 4,271 / 5,832 |

- Flint BF16 serves these trajectories about 1.15 times as fast as the BF16
  4B at one request at a time and 1.27 times at eight, and 5.5 to 6.7 times
  as fast as the BF16 27B. It saturates near 137K tokens/s from 8 concurrent
  requests (12.4 requests/s up to 64).
- The optimized 4B stack is faster than Flint BF16: 161,699 tokens/s
  (13.0 requests/s) at one request at a time with Gigatoken.
- Not matched: the Qwen numbers are from 2026-11-16 (`agent03`, vLLM 0.31,
  the project's two-logit scoring server, another EU-RO-1 B200 host of the
  same CPU type); Flint used the generic completions endpoint on vLLM 0.34.0.
  One session, no repeat on another day.
- Results: `results/flint_ood/agent_bench/` (`summary.json`, per-pass rows).
