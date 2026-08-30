# Kimi K3 training-mixture teacher cache

Date: 2026-08-30. Status: complete training cache. This is an
in-distribution diagnostic on labeled training data, not a held-out benchmark.

## Frozen contract

The cache covers all 8,688 rows in the paper author's final class-balanced
training artifact. The reconstruction and provenance contract are recorded in
`docs/findings/tool_trajectory_teacher_cost_estimate.md`. The ignored rendered
input is `data/tool_trajectory_monitoring/teacher_training/prompts.jsonl`, with
SHA-256
`c06833bc0c5e8893ea45e92ec642a71e751749ee1cd0796eb88f7ab38402763f`.

All requests used `moonshotai/kimi-k3`, temperature zero, reasoning effort
`none`, top-five token logprobs, the prompt-only `Prediction:0|1` interface,
`max_tokens=64`, no provider fallback, prompt template SHA-256
`2418cc55801deead8983d3bde6e35b59603f0080711ce8e12e1d7299487ac128`,
and the declared 5,029-character shared prefix. A row was accepted only when
the response had a terminal binary prediction and logprobs for both literal
decision tokens.

Provider outages required a provenance-separated campaign:

| Provider | Rows | Request-settings SHA-256 | Reported successful-row cost |
| --- | ---: | --- | ---: |
| Makora | 5,071 | `1fc6c9071634495e6f813d0d2040e8761befb0941ee0b8fa6d31a0deb0a54bdd` | `$121.9676` |
| Morph | 305 | `e5d8b6331b49d69e3a89295a73ccccdcca0405ac8d68b72431b1f6012a76b148` | `$8.4499` |
| Fireworks | 3,312 | `b60fa90560524f6ae207b1dc89041feb0fa44f29b6d627ac7d3692599b7482c5` | `$93.8949` |
| **Total** | **8,688** | three audited groups | **`$224.3124`** |

The cache-shard CLI excluded prompt-matched IDs already completed by another
provider, so the three result files are disjoint and their union contains
exactly one row for every requested ID. The final files and SHA-256 digests are:

- Makora: `results/tool_trajectory_monitoring/kimi_k3_training_logits.jsonl`,
  `2ee1a01c8ad91b2e1559a9d032162619e06068802b9b746d2c9d107e7dbbe5e5`;
- Morph: `results/tool_trajectory_monitoring/kimi_k3_training_logits_morph.jsonl`,
  `46b86e18f82fa173d45abcc4dce095fb3fd832517f6cab094f48c9956095ef4d`;
- Fireworks:
  `results/tool_trajectory_monitoring/kimi_k3_training_logits_fireworks.jsonl`,
  `8a29c625b08535f3ea88bd23058473059f4e2003e98d3fe4196a59b926b0378c`.

Do not concatenate these shards without checking duplicate IDs and prompt
hashes. Student preparation should accept the three files as one explicit
provider-stratified cache union and retain each row's provider and request hash.

## Provider agreement and operations

Before each handoff, 64 already-completed rows were rescored on the new
provider. Relative to Makora, Morph had Pearson `0.9981`, Spearman `0.9929`,
mean absolute score delta `0.0118`, maximum delta `0.1244`, one threshold flip,
and identical AUROC `0.9633`. Fireworks had Pearson `0.9947`, Spearman `0.9913`,
mean absolute delta `0.0226`, maximum delta `0.2091`, one threshold flip, and
AUROC `0.9643` versus Makora's `0.9633`. These checks support exploratory use
of the mixed-provider cache but do not prove provider interchangeability.
Provider should remain a stratification variable in student diagnostics.

Makora degraded to about 4 displayed tokens/second and frequent shared-pool
capacity 429s. Morph initially completed 64/64 canary rows in about one minute,
then degraded to `service_overloaded` 429s and upstream 502s even at concurrency
one. Fireworks completed its 64-row canary in 26 seconds and sustained roughly
500 valid rows per minute at concurrency 16. A final provider-specific recovery
pass filled the small number of empty-response and format failures, leaving zero
missing rows.

The two 64-row provider canaries reported `$4.0116` combined successful cost.
Therefore the known successful-request spend including canaries is at least
`$228.3240`. Actual campaign spend is higher because the generic cache still
does not persist usage for paid HTTP-200 responses that contain empty content,
missing logprobs, or invalid terminal output. Do not present `$224.3124` as the
complete account charge.

## Score result

The ignored summary is
`results/tool_trajectory_monitoring/kimi_k3_training_logits_summary.json`, with
SHA-256
`386f7865c0e7b849e8e50a2603fc3c8a134cfce4133a4223196c8a331d0cf721`.

| Population | N | AUROC | pAUROC@20 | AUROC bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: |
| pooled | 8,688 | **0.9673** | **0.8645** | **0.9644--0.9700** |
| Bash Arena | 2,790 | 0.9855 | 0.9476 | 0.9818--0.9890 |
| Bash Bench | 1,436 | 0.9931 | 0.9656 | 0.9899--0.9958 |
| CoT-Red-Handed | 2,340 | 0.9905 | 0.9575 | 0.9876--0.9931 |
| Gloom | 772 | 0.9921 | 0.9670 | 0.9872--0.9960 |
| STRIDE | 1,350 | 0.9959 | 0.9801 | 0.9931--0.9980 |

The pAUROC values use the paper's raw normalized definition,
`integral(TPR dFPR, 0 <= FPR <= 0.2) / 0.2`, rather than scikit-learn's
standardized partial-AUC correction. The unweighted mean source pAUROC@20 is
`0.9636`, which is the training-mixture analogue of the paper's mean-per-dataset
reporting convention. The lower pooled pAUROC@20 of `0.8645` indicates
meaningful cross-source score calibration or ranking differences and should not
be substituted for the macro result without stating the aggregation rule.

The unweighted mean source AUROC is `0.9914`. Pooled balanced accuracy is
`0.9019`, Brier score is `0.0747`, recall is `0.8835`, and FPR is `0.0797` at
the fixed 0.5 threshold. There are 6,458 unique continuous scores. The class
counts remain exactly balanced at 4,344 per class.

This strong result supports binary-logit supervision but is expected to be
optimistic: these are the labeled training sources used to construct the
teacher cache. It must not be described as matching the paper's held-out or OOD
frontier. The held-out ID cache was deliberately deferred on 2026-08-30.

## Cache behavior

Successful rows contain 83,640,053 prompt tokens and 87,010 completion tokens.
Only 1,222,659 prompt tokens were reported as cache reads, a `1.46%` aggregate
fraction, with zero reported cache-write tokens. The provider-specific read
fractions were `1.40%` for Makora, `11.21%` for Morph, and `0.52%` for
Fireworks. Most reuse appeared on exact retries of long prompts, including the
final recovery attempts. The 64-row cold Morph and Fireworks canaries both
reported zero reads. This does not establish cross-row reuse of the declared
teacher prefix and does not justify budgeting future campaigns with shared
prefix savings.

Regenerate the summary with:

```bash
python -m experiments.tool_trajectory_monitoring.summarize_teacher_canary \
  --input data/tool_trajectory_monitoring/teacher_training/prompts.jsonl \
  --scores results/tool_trajectory_monitoring/kimi_k3_training_logits.jsonl \
  --scores results/tool_trajectory_monitoring/kimi_k3_training_logits_morph.jsonl \
  --scores results/tool_trajectory_monitoring/kimi_k3_training_logits_fireworks.jsonl \
  --output results/tool_trajectory_monitoring/kimi_k3_training_logits_summary.json
```
