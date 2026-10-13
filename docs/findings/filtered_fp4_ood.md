# Strict-filtered FP4/FA4 OOD evaluation

On 2026-10-13 the user authorized OOD scoring of the original strict-filtered
FP4 MLP/BF16 FlashQLA/FA4 adapter, selecting BF16 serving only. The frozen
[experiment contract](../../experiments/filtered_fp4_ood/README.md) evaluates
two fixed passes over all 6,395 unchanged examples against cached filtered
SDPA, released Gleipnir 4B and unfiltered references. Repeated use of this OOD
suite and missing assistant prose remain disclosed; there is no selection or
promotion on OOD results.

The FP32 master SHA256 is
`4eb97e5385f53b135c0b77cb92d0cbbfa78a661100f9ba8ac4205dbabb34f057`.
Its original FP32 inference export reconstructs the original BF16 merged model
with exact weight, config and tokenizer hashes. Three fresh serving canaries
and the BF16 native dispatch audit pass. Both passes completed on the existing
NC2 B200. All 6,395 predictions per pass were collected locally and the complete
ranking/calibration metrics independently recomputed exactly from saved scores.
Artifacts live under `results/filtered-fp4-ood01/` and logs under
`logs/runpod/filtered-fp4-ood01/`. Capacity remains running.

| Model/training recipe, BF16 serving | Source-macro pAUROC@20 | Source-macro AUROC |
| --- | ---: | ---: |
| Strict-filtered FP4 MLP/FA4, mean of two passes | 0.737928 | 0.899799 |
| Cached strict-filtered BF16/SDPA | 0.786936 | 0.921876 |
| Cached released Gleipnir 4B | 0.782350 | 0.908584 |
| Cached unfiltered regular | 0.776238 | 0.914348 |

The FP4/FA4 model loses 0.049007 source-macro pAUROC against the strict-filtered
SDPA control. Per-source pAUROC values are the fixed two-pass mean:

| Source | FP4/FA4 | Filtered SDPA |
| --- | ---: | ---: |
| AgentDojo | 0.891788 | 0.929880 |
| CUA SHADE | 0.735926 | 0.771191 |
| IAC | 0.730427 | 0.813657 |
| MLE sabotage | 1.000000 | 1.000000 |
| Rogue eval | 0.891374 | 0.969717 |
| SHADE arena | 0.178057 | 0.237170 |

The drop spans five sources, with the largest differences on IAC and Rogue eval.
This is evidence of worse transfer for this fixed FP4/FA4 checkpoint, despite its
training speed advantage. Historical seed/runtime/host differences and the
combined precision/attention intervention prevent attributing the gap to FP4
alone. BF16 serving isolates this from optimized deployment precision drift.
No OOD promotion or recipe reversal is inferred automatically.

The two pAUROC results are 0.738005 and 0.737852, only 0.000154 apart. Score
MAE between repeats is 0.001473, correlation 0.999869, maximum difference
0.062177, with six threshold-0.5 flips. Scoring takes 497.49 and 496.96 seconds
(about 8.3 minutes each), with 55,346,879 prompt tokens per pass and roughly
111k input tokens/s. Repeat variation is much smaller than the control gap.
