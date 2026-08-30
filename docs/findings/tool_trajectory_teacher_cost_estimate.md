# Kimi K3 teacher-cache cost estimates

Date: 2026-08-30. This estimate prices annotation only; it does not authorize a
paid launch.

## Selected paper mixture

Use the paper author's final 8,688-row artifact
`adityaasinha28/deliberative-monitor-pipeline` at revision
`2f10f4e49f4bf349965640ac021406689455d1c3`, file
`merged/stride_gloom_crh_s45_o46_ctrl_t7_bal.parquet`. Its SHA-256 is
`43bfc9f78d6584c43135109636f5e1747c694842419a327c3f37614fb9301802`.
The upstream card does not declare a license, so keep the materialization for
internal research and do not redistribute it.

The artifact has the paper's exact class-balanced source counts. Its BashArena
component consists of 1,770 Sonnet 4.5 rows and 1,020 Opus 4.6 rows:

| Source | Benign | Positive | Rows |
| --- | ---: | ---: | ---: |
| STRIDE | 675 | 675 | 1,350 |
| Gloom | 386 | 386 | 772 |
| CoT-Red-Handed | 1,170 | 1,170 | 2,340 |
| Bash Arena | 1,395 | 1,395 | 2,790 |
| Bash Bench | 718 | 718 | 1,436 |
| **Total** | **4,344** | **4,344** | **8,688** |

Reapplying the paper's 28,000-token input rule with `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218` retains all 8,688 rows. The maximum
action-only input is 27,945 tokens. This confirms the released final artifact
is already filtered and no additional sampling or replacement is needed.

Rendering the frozen full teacher prompt and tokenizing it with
`moonshotai/Kimi-K3` revision
`a590ce090cb049c93a33dfe8c208ec652aa20503` produces 83,457,605 raw prompt
tokens. Adding the 21 chat-wrapper tokens observed exactly on paid Makora
canary rows gives **83,640,053 predicted billable prompt tokens**. Mean prompt
length is 9,627.1 provider tokens; p50 is 7,274, p90 is 22,445.8, p95 is
24,863.7, p99 is 26,490.5, and the maximum is 28,331.

| Source | Rows | Predicted Makora prompt tokens | Cost with 10 output tokens |
| --- | ---: | ---: | ---: |
| STRIDE | 1,350 | 7,867,477 | `$20.23` |
| Gloom | 772 | 13,026,767 | `$33.32` |
| CoT-Red-Handed | 2,340 | 3,060,735 | `$8.10` |
| Bash Arena | 2,790 | 48,358,605 | `$123.67` |
| Bash Bench | 1,436 | 11,326,469 | `$29.07` |
| **Total** | **8,688** | **83,640,053** | **`$214.39`** |

At the recorded Makora rates, the total is `$213.2821` input plus `$1.1077`
for ten output tokens per row. Continue to budget without cross-row cache
savings: the invariant prefix is below Makora's observed shared-prefix cache
threshold, and the paid canary reported no reuse. A 10% authorization envelope
is `$235.83`; re-query endpoint prices and fail closed before a paid launch.

The ignored, prompt-aware cache input and manifest are reconstructed with:

```bash
python -m experiments.tool_trajectory_monitoring.prepare_teacher_training_cache
```

The manifest records the source checksum and revision, exact source/label
counts, tokenizer revisions, 28K filter result, prompt hashes, per-row token
counts, and aggregate/per-source distributions. Its current output SHA-256 is
`c06833bc0c5e8893ea45e92ec642a71e751749ee1cd0796eb88f7ab38402763f`.

## Historical accessible-release estimate

> **Scope warning:** this is **not** the paper's 8,688-row training mixture. It
> prices an earlier/different 9,500-row gated release that was locally
> accessible. Do not use `$184.33` as the exact budget for a paper-data
> replication.

### Materialized pool and counting method

The accessible `Syghmon/blackboxmonitorsMATS` revision
`ccb77e37589baaee1f0e16ac9ab89ceac8f6cc8a` contains 9,500 rows:

- 7,500 `control_arena`;
- 1,000 `gemini_synthetic`;
- 1,000 `unified_dataset`.

Applying the paper's 28,000-token trajectory filter with
`Qwen/Qwen3.5-9B` tokenizer revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` leaves 8,538 rows and filters
962. This was a comparability assumption, not a Qwen3.5-9B context limit. The
paper used 28K to reserve 4K for rationale completions in a 32K training
context; our one-step logit target does not require that output headroom, and
Qwen3.5-9B natively supports a much longer context. We should therefore retain
or replace this filter only as an explicit experiment/data-contract decision.

The calculation renders the unchanged teacher prompt around every eligible
trajectory and tokenizes it with the official `moonshotai/Kimi-K3` tokenizer at
revision `a590ce090cb049c93a33dfe8c208ec652aa20503`.

Local counts were calibrated against paid Makora responses. On every checked
canary row, Makora prompt usage was exactly the official Kimi raw-prompt token
count plus 21 chat-wrapper tokens. This gives 71,860,759 predicted provider
prompt tokens across the eligible release. Mean prompt length is 8,416.6 Kimi
tokens; p50 is 5,660.5, p90 is 19,434.5, p95 is 22,074.4, p99 is 25,175.6, and
the maximum is 28,472.

### Cost at current Makora rates

The observed OpenRouter rates are `$2.55/M` ordinary prompt tokens,
`$0.256/M` cache-read tokens, and `$12.75/M` completion tokens.

| Source | Eligible / raw rows | Prompt tokens | Cost with 10 output tokens |
| --- | ---: | ---: | ---: |
| Control Arena | 6,721 / 7,500 | 53,713,317 | `$137.83` |
| Gemini synthetic | 1,000 / 1,000 | 4,691,062 | `$12.09` |
| Unified synthetic | 817 / 1,000 | 13,456,380 | `$34.42` |
| **Total** | **8,538 / 9,500** | **71,860,759** | **`$184.33`** |

The total consists of `$183.2449` input and `$1.0886` output. Ten output tokens
per row use the completed 64-token-recovery responses as the conservative
normal-completion estimate; setting the cap to 64 does not itself bill 64
tokens. The original eight-token cap would price at `$184.12` before recovery.
The tested unconstrained one-token scalar candidate would price at `$183.81`;
the interface choice therefore changes the full bill by less than one dollar.

The 921-token invariant teacher prefix is below Makora's observed roughly
2,000-token cross-row cache threshold. The budget consequently assumes no
cross-row cache reads, matching the 100-row canary. If every warm row could
cache that prefix, the theoretical saving would be only `$18.04`, but this was
not observed and must not be included in the launch budget. Exact retries can
still be cheaper when the complete prompt remains in cache.

Use `$205` as a practical authorization envelope for this 8,538-row release:
it is approximately 11% above the modeled `$184.33` and covers price drift and
bounded retries. Re-query endpoint prices immediately before launch and fail
closed if they exceed the recorded Makora ceiling.

### Superseded paper-count projection

The earlier row-count projection was `$187.57`. It is superseded by the exact
`$214.39` count above. The underestimate came primarily from the much longer
BashArena component, which contributes 57.8% of billable prompt tokens.
