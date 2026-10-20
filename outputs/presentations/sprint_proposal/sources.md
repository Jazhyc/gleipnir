# Sources

Numbers are from the
[OOD frontier registry](../../../docs/research/tool_trajectory_ood_frontier.md)
at revision `9e45f8a` (registry SHA-256 `5a23ac27…164c6`).

- **Metric.** Unweighted mean over six OOD sources of the paper's raw normalized
  pAUROC@20 on the strict 6,395-row suite of Sinha et al. (2026),
  [arXiv:2605.29601](https://arxiv.org/abs/2605.29601).
- **Cost.** Uncached USD per 1,000 evaluations. Gleipnir models are priced at
  self-hosted break-even on one fully utilized B200 running optimized serving:
  `$7.99/h` on-demand (Runpod) or `$3.44/h` committed-use (marked `*`).
  Other rows use list prices, which include provider margin. Training cost is
  excluded.
- **Headline ratio.** Gemini 3.1 Pro prompted (0.940, `$73.218`) against
  Gleipnir 27B (0.9508, `$0.3896` on-demand / `$0.1677` committed): 188× / 437×.
- **Swarm example.** One million evaluations at the above rates: `$390` against
  `$73,218`, for trajectories of the suite's mean length. The figure is illustrative only.
- **Gleipnir 4B.** v0.3 seed average with QAD, optimized serving: 0.7703 at
  `$0.0850` (`$0.0366` committed). Compared with the paper's Qwen3.5-4B SFT+RL:
  0.684 at `$0.358`.
- **Figure.** `assets/ood_frontier.pdf` was rendered with
  `python scripts/plot_tool_trajectory_ood_frontier.py --output
  outputs/presentations/sprint_proposal/assets/ood_frontier.pdf`. It is the same
  simplified figure as `figures/tool_trajectory_ood_frontier.svg`.
