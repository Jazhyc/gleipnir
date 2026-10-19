# pplx-decider v1.2 native-readout OOD benchmark

Evaluation: 2026-10-19 on a Runpod B200 (`GPU-58276fb0…`), at the user's
request. Design, gates, deviations and per-source tables are in the
[experiment record](../../experiments/pplx_decider_ood/README.md).

## Result

Both open checkpoints were scored locally on all 6,395 frozen strict-OOD rows,
with the exact state and Noul question of the
[Jev 1.13 benchmark](tool_trajectory_jev_ood_benchmark.md).

| Model | Backbone | Mean-OOD pAUROC@20 | Mean-OOD AUROC | Cost per 1K |
|---|---|---:|---:|---:|
| `pplx-decider-v1.2` | Qwen3.8-27B | **0.895583** | 0.957681 | **$0.176261** |
| `pplx-decider-flash-v1.2` | Qwen3.5-4B | 0.534976 | 0.832991 | $0.088131 |
| `pplx-decider-flash-v1.2`, teacher-spec question | Qwen3.5-4B | 0.540751 | 0.838150 | $0.095991 |

- **The 27B enters the frontier.** It dominates Gleipnir 4B and Jev 1.13, and
  beats Jev on every source.
- **Flash is not added.** At the user's request, it is excluded from the
  registry and plot because it falls below base Qwen3.5-4B under the full
  binary specification (0.6175).
- **Interface.** The state is the raw trajectory and the question is the
  frozen Noul question. These are configuration comparisons, not
  prompt-matched model comparisons.

## Interface and identities

- **Runtime.** Perplexity's `pplx_decider` 0.3.0 (Transformers backbone,
  non-causal full attention, 255-way BF16 readout), not vLLM. The pinned
  stack is torch 2.14.0+cu130, Transformers 5.17.0 and FLA 0.5.2. The
  release requirements omit `causal_conv1d`, so its PyTorch fallback runs.
  Checkpoint files match the release manifest; only `README.md` differs.
- **Revisions.** 27B `cce0f3f80033e9d89d80831d5281fd72124aacce`; Flash
  `9ff7cf9838e59df461c276b7699d56b244d2af58`.
- **Scores.** Ranking uses the raw readout margin `true − false`. Threshold
  and Brier use `sigmoid(margin / T)` with the saved temperature (1.382 for
  the 27B, 1.554 for Flash). The calibrated probabilities show heavy
  under-flagging at 0.5 for Flash (recall 0.12) and moderate under-flagging
  for the 27B (0.70).
- **Path.** Packed fast path with `torch.compile` and CUDA graphs. Flash
  missed the frozen fast-path parity gate by 0.00008 in probability
  (BF16 readout steps). The user then chose the fast path, the way the models
  are used in practice. Fast and reference paths agree on OOD rows scored by
  both: Flash 1,470 rows, AUROC 0.81098 vs 0.81095; 27B 264 rows, margin
  Pearson 0.998. Token counts are identical.
- **Cost.** The local joint-sequence input count times OpenRouter list
  prices for the `-20261016` endpoints: $0.02/M for the 27B, $0.01/M for
  Flash, no output charge and no caching. Both models produce identical
  counts: 56,359,499 tokens, 8,813.06 per row, longest 29,723. Hosted
  serving precision and billing are not observed; this reproduces the
  published weights, not the API.
- **Inputs.** States `c38c6214…` (6,395 rows); question `9b423177…`; config
  `b4514d2c…`.
- **Scores (ignored artifacts).** `results/pplx_decider_ood/decider/full_fast.jsonl`
  `319f2afe…`; `flash/full_fast.jsonl` `5c28ae63…`;
  `flash/teacher_full_fast.jsonl` `1b49779d…`.
- **Selection.** Nothing was selected on OOD: one run per arm, the frozen
  question, the saved temperature and every row.
