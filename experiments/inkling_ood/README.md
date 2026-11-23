# Inkling 2 Mini on the strict OOD suite

Could [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
succeed Qwen3.8-27B as the base for the large monitor? Inkling 2 Mini is a
26.4B looped mixture-of-experts model (Apache 2.0, released 2026-11-20,
revision `7a67ae23…`): 24 layers, of which the middle 12 run three times with
shared weights, so 3.4B active parameters per token and about 5.4B effective.
The pass count is fixed (`loop_config.num_passes: 3`, `is_varloop: false`;
vLLM rejects `loop_num_passes` overrides on this checkpoint). The only
checkpoint is mixed NVFP4/MXFP8 (about 20 GB); there is no BF16 release.

This experiment scores the unadapted checkpoint under the contract that chose
Qwen3.8-27B ([Qwen 27B generations](../qwen27b_generations_ood/README.md)) and
used for [Flint](../flint_ood/README.md), and measures its throughput.

## Pre-registration (frozen 2026-11-23, before any scoring)

- **Hypothesis.** Unadapted Inkling 2 Mini reaches Mean-OOD pAUROC@20 within
  0.02 of unadapted Qwen3.8-27B (0.8901) at the direct-logit interface.
- **Intervention.** The checkpoint, and what it forces: its chat template and
  tokenizer, its native NVFP4/MXFP8 weights, and a vLLM nightly (support
  merged on 2026-11-21 in vllm-project/vllm#69028, after v0.34.0; the exact
  build is recorded in the configs). Unchanged from the Qwen 27B contract:
  the 6,395 rows, the full teacher prompt, the `Prediction:` prefix, literal
  `0`/`1` logprob scoring, 16 sequences, 32,768-token batches and seed
  20260830.
- **Decision boundary.** The template has no generation-prompt branch and no
  thinking switch. It writes `reasoning_effort` as a system line
  (`Thinking effort level: <x>`) before the user turn and stops after the
  user turn; the card warns that the minimum effort "does not fully disable
  reasoning". The prompt continues with the model's own text-answer opening,
  `<|message_model|><|content_text|>Prediction:`, so the scored turn carries
  no reasoning. The canary records, as a diagnostic only, the unconstrained
  token after the bare `<|message_model|>` (`<|content_thinking|>` or
  `<|content_text|>`).
- **Arms, both frozen now.** Effort changes only the system line, not the
  loop count or compute.
  - **A, primary (`inkling_effort_none.json`):** `reasoning_effort="none"`
    (renders as level 0).
  - **B (`inkling_effort_high.json`):** `reasoning_effort="high"` (level 0.9,
    the template's default).
  Both are reported whatever they show.
- **Engine settings that depart from the Qwen contract.** Maximum length is
  40,960 instead of 32,768: Inkling's tokenizer makes the longest prompt
  37,998 tokens, and 47 prompts (46 IaC, 1 CUA-SHADE-Arena) exceed 32,768.
  Nothing is truncated; long prompts prefill in chunks. Prefix caching is
  off (it changed Flint's logits on vLLM 0.34; Inkling's per-pass caches are
  new code; Flint's throughput was measured without it). Speculative decoding
  (the bundled DSpark drafter) and the reasoning parser are not used. Image
  inputs are disabled (`limit_mm_per_prompt={"image": 0}`). KV block size is
  the runtime default.
- **Baselines.** Qwen3.8-27B 0.8901, Qwen3.6-27B 0.8726, Qwen3.5-27B 0.8388,
  unadapted Qwen3.5-4B 0.6175, Flint 0.4486 (bare) and 0.4811 (empty think).
  The Qwen and Flint baselines are BF16; Inkling is scored at its released
  low precision.
- **Reporting.** Mean-OOD pAUROC@20 (unweighted mean of six sources),
  per-source pAUROC@20 and AUROC, macro Brier, balanced accuracy, score ties;
  scoring seconds, rows/s and prompt tokens/s.
- **Reading rule.** Descriptive. "Competitive" means Mean-OOD pAUROC@20 of at
  least 0.8701. The user decides any follow-up (ID scoring, adapter training).
  Nothing is tuned on these labels.
- **Stop conditions.** Input, manifest, revision, teacher-prompt, rendering or
  boundary drift; truncation or a token-count mismatch; a missing decision
  logprob; a 13-row canary (one row per source and label, plus the longest
  prompt) that is incomplete, non-finite, or whose batched and singleton
  scores differ by more than 0.02 on average or 0.05 at most; missing or
  duplicate rows.
- **Throughput, after quality.** The 96 agent trajectories at 1 and 8
  concurrent requests (as in [Flint](../flint_ood/README.md#serving-throughput-on-the-96-agent-trajectories-agent_benchpy)),
  served by stock `vllm serve` from the same nightly.
- **Compute.** User authorization 2026-11-23: one B200 Pod, kept running
  afterwards for follow-up experiments. EU-RO-1 had no B200, so the Pod is in
  US-TX-6; throughput is therefore compared across hosts.

### Amendment (2026-11-23, after the canary, before any full-suite score)

The vLLM runtime is the nightly `0.34.0rc1.dev1201+gbfe3d7e87`, the first
after the support merge; the stable 0.34.0 lacks `Inkling2LoopForConditionalGeneration`.

The arm A canary (16 sequences) stopped on the batched-versus-singleton gate:
mean 0.030, maximum 0.381 in score. `diagnose_parity.py` located the cause
without reading labels (`results/inkling_ood/diagnostics/`). With prefix
caching off, the same batch scored twice and the same singletons scored twice
are bit-identical, but batch composition moves margins by 0.83 nats on
average and 5.25 at most (a reversed batch 5.48). Five of the 13 prompts move;
eight are identical. Turning CUDA graphs off does not remove it (0.71 mean,
4.93 maximum), and singleton margins themselves differ by 0.10 nats on average
(1.04 at most) between graph and eager mode. A run with a 65,536-token batch
budget, to rule out chunked prefill, failed at engine start-up (Triton
`invalid argument`). Both configs now set `max_num_seqs: 1` (user decision),
so each prompt is scored alone: deterministic, and the canary gate compares
like with like. Throughput from this run therefore measures one request at a
time. The failed canary is kept as
`results/inkling_ood/inkling_effort_none_canary_batched_failed/`.

The canary also showed that the direct boundary is one Inkling does not use:
after the bare `<|message_model|>`, `<|content_thinking|>` has probability
about 1 on all 13 prompts at effort 0, and `<|content_text|>` is 22 to 28 nats
below it.

## Commands

```bash
PYTHONPATH=src:. python -m experiments.inkling_ood.audit none   # tokenizer only
PYTHONPATH=src:. python -m experiments.inkling_ood.audit high
# on the Pod, after syncing the code and the OOD input
experiments/inkling_ood/setup.sh
CANARY_ONLY=1 experiments/inkling_ood/run.sh inkling_effort_none
experiments/inkling_ood/run.sh inkling_effort_none
experiments/inkling_ood/run.sh inkling_effort_high
# consistency diagnostics behind the amendment
python -m experiments.inkling_ood.diagnose_parity --output results/inkling_ood/diagnostics/parity_default.json
python -m experiments.inkling_ood.diagnose_parity --enforce-eager --output results/inkling_ood/diagnostics/parity_eager.json
python -m experiments.inkling_ood.agent_bench --output results/inkling_ood/agent_bench
PYTHONPATH=src:. python -m experiments.inkling_ood.compare   # on the VPS
```

## Interface audit (`audit.py`, CPU only)

| | Inkling, effort none | Inkling, effort high | Flint | Qwen 27B |
|---|---:|---:|---:|---:|
| prompt tokens, total | 65,608,412 | 65,621,202 | 50,935,936 | 60,539,619 |
| prompt tokens, maximum | 37,996 | 37,998 | 26,047 | 30,382 |
| decision tokens `0` / `1` | 15 / 16 | 15 / 16 | 15 / 16 | 15 / 16 |

- Inkling's tokenizer needs 8% more tokens than Qwen's for the same prompts.
- The boundary tokenizes as `<|end_message|>`, `<|message_model|>`,
  `<|content_text|>`, `Prediction`, `:`, then the digit. Neither the template
  nor the tokenizer adds a BOS token.
- The template does not sanitize user text. All 1,499 Rogue Eval prompts
  contain literal `<|endoftext|>` (5,444 times), which the tokenizer reads as
  Inkling's pad token, as Qwen's tokenizer reads it as a special token too.
- Rendered-prompt SHA-256: `d7a9d95f…` (none), `da56fb1f…` (high).

## Results (2026-11-23, US-TX-6 B200 `GPU-62b8cec6…`, one sequence at a time)

Pod `5uo9tcfq79m724` (Xeon 6972P, driver 595.112.07), vLLM
`0.34.0rc1.dev1201+gbfe3d7e87`, torch 2.14.0+cu130. Both arms scored all
6,395 rows, passed the canary (batched and singleton scores identical) and
hit no stop condition. Engine start-up took 145 s with warm caches; the first
start spent about 12 minutes on one-off kernel builds and compilation.

**Unadapted Inkling 2 Mini is competitive with Qwen3.8-27B.** The primary arm
is 0.6 points below it and 27 points above unadapted Qwen3.5-4B.

| strict OOD | effort none | effort high | Qwen3.8-27B | Qwen3.5-4B, unadapted | Flint, empty think |
|---|---:|---:|---:|---:|---:|
| **Mean-OOD pAUROC@20** | **0.8842** | 0.8802 | **0.8901** | 0.6175 | 0.4811 |
| macro AUROC | 0.9452 | 0.9444 | **0.9535** | 0.8549 | 0.8025 |
| macro Brier | 0.1041 | 0.1098 | **0.0823** | | 0.2356 |
| macro balanced accuracy | 0.8812 | 0.8730 | **0.8894** | | 0.6534 |
| tied rows | 101 | 87 | 2,424 | | 3,140 |

| pAUROC@20 (AUROC) by source | effort none | effort high | Qwen3.8-27B |
|---|---:|---:|---:|
| AgentDojo | **0.9979** (0.9996) | 0.9895 (0.9979) | 0.9657 (0.9931) |
| CUA-SHADE-Arena | 0.8390 (0.9444) | 0.8318 (0.9457) | **0.8648** (0.9322) |
| IaC | **0.9865** (0.9965) | 0.9802 (0.9952) | 0.9678 (0.9921) |
| MLE-Sabotage | 0.9827 (0.9965) | 0.9898 (0.9980) | **1.0000** (1.0000) |
| Rogue Eval | 0.9858 (0.9972) | **0.9887** (0.9977) | 0.9862 (0.9972) |
| SHADE-Arena | 0.5134 (0.7373) | 0.5012 (0.7317) | **0.5560** (0.8062) |

- **Inkling leads on AgentDojo and IaC** and trails on CUA-SHADE-Arena,
  MLE-Sabotage and SHADE-Arena; SHADE-Arena remains every model's weakest
  source, and Inkling's AUROC there is 7 points below Qwen's.
- **Calibration is weaker** (Brier 0.104 vs 0.082), and Inkling's scores
  correlate with Qwen3.8-27B's at 0.87 (655 rows on opposite sides of 0.5).
  Its scores are nearly untied (101 tied rows against Qwen's 2,424).
- **Effort barely matters.** The two arms' scores correlate at 0.97 (224 rows
  flip at 0.5); effort none is 0.4 points higher.
- **The boundary is still one Inkling does not use.** At both efforts the
  unconstrained token after `<|message_model|>` is `<|content_thinking|>`,
  with `<|content_text|>` 21 to 28 nats below. Unlike Flint, the unadapted
  model ranks well at this boundary anyway.

**Scoring throughput** (one sequence at a time, own tokenizer):

| scoring the 6,395 rows | effort none | effort high | Flint | Qwen3.8-27B |
|---|---:|---:|---:|---:|
| seconds | 1,324 | 941 | 365 | 2,190 |
| rows/s | 4.8 | 6.8 | 17.5 | 2.9 |
| prompt tokens/s | 49,500 | 69,700 | 139,600 | 27,600 |

- The two Inkling arms ran the same rows on the same Pod, yet arm A was
  uniformly about 1.4 times slower on the long-prompt batches. Their prompts
  differ by two tokens, so this is run-to-run variation on this host, not a
  property of the effort level.
- Not matched: Flint and Qwen used up to 16 sequences, other hosts (EU-RO-1
  EPYC 9555) and other vLLM versions; Qwen also used prefix caching.

## Serving throughput on the 96 agent trajectories (`agent_bench.py`)

As for [Flint](../flint_ood/README.md#serving-throughput-on-the-96-agent-trajectories-agent_benchpy),
at the report's levels only (8 and 1; three passes each after one warmup pass
at 8): stock `vllm serve` from the same nightly, no prefix caching, 40,960
maximum length, 128 sequences, effort none boundary, `/v1/completions` with
one constrained token. Inkling tokenizes these rows to 1,394,704 tokens
(median 11,364) against Qwen's 1,193,325, so requests/s is the like-for-like
unit. At 8 concurrent requests the server batches, so its scores are not
batch-invariant (see the amendment); they are timing payloads only.

| | Inkling | Flint BF16 | Qwen3.5-4B BF16 | Qwen3.8-27B BF16 |
|---|---:|---:|---:|---:|
| requests/s, one at a time | 4.3 | 9.6 | 8.3 | 1.7 |
| requests/s, 8 concurrent | 5.6 | 12.3 | 9.7 | 1.8 |
| prompt tokens/s, one at a time | 62,835 | 106,319 | 103,266 | 21,720 |
| prompt tokens/s, 8 concurrent | 80,990 | 136,701 | 120,284 | 22,950 |
| latency p50 / p95, one at a time, ms | 184 / 490 | 82 / 226 | 94 / 267 | 452 / 1,281 |
| latency p50 / p95, 8 concurrent, ms | 1,456 / 2,033 | 627 / 907 | 816 / 1,131 | 4,271 / 5,832 |

- Inkling serves these trajectories 2.5 to 3.1 times as fast as the BF16 27B
  and at about half the rate of the BF16 4B.
- Not matched: the other columns are from EU-RO-1 B200 hosts (Qwen on vLLM
  0.31 with the project's scoring server, 2026-11-16; Flint on vLLM 0.34.0,
  2026-11-20). One session, no repeat on another day.

## Where the GPU time goes (`profile_kernels.py`)

User request, 2026-11-23: Inkling needs about 5 times fewer linear-layer
FLOPs per token than the 27B (about 5.4B effective active parameters against
27B) but scores only about 1.2 times as many tokens/s as the optimized 27B,
so where does the time go? Four of the 96 agent trajectories, rendered at the
effort none boundary, scored alone on the OOD engine settings: five timed
requests each, then one torch-profiler trace each
(`results/inkling_ood/profile/`).

| prompt tokens | median ms | prompt tokens/s | GPU busy in trace, ms |
|---:|---:|---:|---:|
| 1,760 | 36.9 | 47,700 | 32.2 |
| 8,161 | 121.1 | 67,400 | 104.9 |
| 16,420 | 240.5 | 68,300 | 201.1 |
| 33,988 | 493.8 | 68,800 | 454.4 |

A first run, stopped by a path bug after its timing phase, measured
70,600 to 74,100 tokens/s from 8K to 34K tokens
(`profile_failed_path/`), the same host variation as between the two OOD
arms.

Share of GPU kernel time (the 30 largest kernels cover 98 to 100%):

| kernel family | 1.8K | 8.2K | 16.4K | 34.0K |
|---|---:|---:|---:|---:|
| FA4 attention core (`FlashAttentionForwardSm100`, BF16) | 7% | 10% | 13% | 17% |
| attention glue: relative logits `_qr_rel`, `_attn_res`, `_attn_prologue`, `_gate_sink` | 17% | 23% | 21% | 21% |
| dense BF16 GEMMs (`nvjet`: attention projections, LM head) | 17% | 17% | 17% | 15% |
| expert GEMMs (NVFP4 and MXFP8 `bmm`) | 20% | 14% | 14% | 13% |
| MoE activation quantization, routing, scatter, combine | 16% | 14% | 14% | 13% |
| RMS norms (`_blend_rms_norm`, `_pair_rms_norm`) | 8% | 11% | 11% | 11% |
| short convolutions (`convMixKernel`) | 8% | 10% | 10% | 9% |

- **The GPU is never idle.** Kernel time fills the whole request (busy time
  equals or exceeds the span because some kernels overlap), so CPU or launch
  overhead is not what holds it back.
- **The FP4 expert GEMMs are only 13 to 14%** of the time at 8K tokens and
  above. Quantizing activations and moving tokens to and from the experts
  costs as much again. The MoE design saves FLOPs, but FLOPs are not where
  the time goes.
- **About half the time is memory-bound Triton glue**: attention prologue and
  relative logits, the cross-layer attention residual, gates and sinks, RMS
  norms, short convolutions and MoE quantization. On a 2,304-wide hidden
  state these move data more than they compute.
- **The BF16 attention projections cost more than the FP4 experts** (15 to
  17% against 13 to 14%).
- **Attention scales gently.** Tokens/s is flat from 8K to 34K tokens; the
  FA4 core grows from 10% to 17%.
- **The loop multiplies all of it.** Every family above runs per layer visit,
  and 36 of the 48 visits per token are the three passes of the 12 core
  layers. A single pass would make 24 visits, so the loop roughly doubles the
  per-token layer time. That is the architecture and stays.

**Where speed could come from**, in order of likely size (none tried):
fusing the glue kernels around attention and the MoE (norms, prologue,
relative logits, gates, quantize-and-scatter); FP8 or FP4 attention
projections (W16A16 now); FP8 attention and KV cache at long context; and
fixing batch-invariance so several prompts can share a step. The glue is
about half of GPU time, so fusing it well could plausibly give 1.3 to 1.5
times; this is an estimate, not a measurement.

**Limits.** One unadapted checkpoint at one prompt interface, scored one
sequence at a time because batched scores depend on batch composition on
this vLLM nightly; whether singleton scores match the reference
implementation is unverified (they move by up to 1 nat between graph and
eager mode). ID was not scored.

Artifacts: `results/inkling_ood/` (both arms, `comparison.json`, the audits,
`diagnostics/`, `agent_bench/`, the failed batched canary) and
`logs/runpod/inkling_ood/` (34 files verified against the Pod by SHA-256,
`collect.sha256`). Pod lifecycle:
[infrastructure](../../docs/infrastructure.md#runpod).
