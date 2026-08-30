# Kimi K3 OOD binary-logit canary

Date: 2026-08-30. Status: completed 600-row canary; identical full-suite
evaluation authorized and ready to scale.

## Frozen design

The canary measures Kimi K3 directly on the six strict OOD action-only sources
using Gleipnir's full binary teacher prompt and normalized literal `0` versus
`1` decision-token logprobs. It is an interface and quality canary before the
complete 6,395-row evaluation, not a surface for prompt, threshold, or request
setting selection.

The sample contains 50 benign and 50 positive rows from each source, selected
by a deterministic SHA-256 ordering within every source-label stratum using
seed `20260830`. The 600 rendered inputs have SHA-256
`ce65b1e13adfecce6c73313bcc55fbf839f9679e4acbf39fcf9490d2930781e2`;
their manifest has SHA-256
`77f8ac7a85cb48938ed0299966c38cc0dacf1bc5e21997b46d41d0d3e7aed2f2`.

Every call used:

- model `moonshotai/kimi-k3`, provider `Fireworks`, and no fallback;
- temperature zero, reasoning effort `none`, and top-five token logprobs;
- the prompt-only `Prediction:0|1` interface with `max_tokens=64`;
- price ceilings of `$3.00/M` prompt and `$15.00/M` completion tokens;
- prompt template SHA-256
  `2418cc55801deead8983d3bde6e35b59603f0080711ce8e12e1d7299487ac128`;
- request-settings SHA-256
  `b60fa90560524f6ae207b1dc89041feb0fa44f29b6d627ac7d3692599b7482c5`.

## Result

| Population | N | AUROC | pAUROC@20 |
| --- | ---: | ---: | ---: |
| Mean per source | 6 sources | **0.9632** | **0.9191** |
| Pooled | 600 | **0.9749** | **0.9345** |
| AgentDojo | 100 | 1.0000 | 1.0000 |
| CUA-SHADE-Arena | 100 | 0.9172 | 0.8760 |
| IaC | 100 | 0.9980 | 0.9900 |
| MLE-Sabotage | 100 | 1.0000 | 1.0000 |
| Rogue Eval | 100 | 0.9996 | 0.9980 |
| SHADE-Arena | 100 | 0.8644 | 0.6505 |

The pooled source-label-stratified bootstrap 95% interval for AUROC is
`0.9612--0.9863`. At the fixed 0.5 threshold, pooled balanced accuracy is
`0.9367`, recall is `0.9133`, FPR is `0.0400`, and Brier score is `0.04882`.
There are 555 unique continuous scores among 600 rows.

All 600 rows returned valid logprobs for both literal decision tokens from
Fireworks. The first concurrency-16 pass encountered eight terminal shared-pool
429s after bounded retries; an identical concurrency-2 resume recovered all
eight. There were no format, parse, price-cap, model, provider, or logprob
failures. The completed score cache has SHA-256
`c268958b594e4482f539ccaeb371454f629faeff1f9af96064a69d8bc99003bb`.

Successful rows report 5,498,543 prompt tokens, 6,000 completion tokens, 11,630
cache-read tokens, and `$16.38869` cost. The `0.21%` cache-read fraction is
negligible and does not justify a discounted full-suite budget. The complete
summary has SHA-256
`a195dec46f2df98f87a519d57ac983d580c307e7a706dedbb1d86a903e869c48`.

## Interpretation and scale-up rule

The sample supports scaling the identical Kimi interface to the remaining OOD
rows. Ranking is excellent on four sources and strong on CUA-SHADE-Arena.
SHADE-Arena is the clear hard source, especially in the low-FPR region; do not
change the prompt or threshold in response. The full evaluation must exclude
the 600 completed IDs from its new request shard, then combine both caches for
the frozen 6,395-row summary.

The ignored artifacts are under
`data/tool_trajectory_monitoring/teacher_ood_canary/` and
`results/tool_trajectory_monitoring/`. Reconstruct and summarize with:

```bash
python -m experiments.tool_trajectory_monitoring.prepare_teacher_ood_canary
python -m experiments.tool_trajectory_monitoring.summarize_teacher_canary \
  --input data/tool_trajectory_monitoring/teacher_ood_canary/prompts.jsonl \
  --scores results/tool_trajectory_monitoring/kimi_k3_ood_canary_fireworks.jsonl \
  --output results/tool_trajectory_monitoring/kimi_k3_ood_canary_summary.json
```
