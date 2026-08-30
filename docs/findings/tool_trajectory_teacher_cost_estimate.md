# Kimi K3 full-pool teacher-cache cost estimate

Date: 2026-08-30. This estimate prices annotation only; it does not authorize a
paid launch.

## Materialized pool and counting method

The accessible `Syghmon/blackboxmonitorsMATS` revision
`ccb77e37589baaee1f0e16ac9ab89ceac8f6cc8a` contains 9,500 rows:

- 7,500 `control_arena`;
- 1,000 `gemini_synthetic`;
- 1,000 `unified_dataset`.

Applying the frozen 28,000-token trajectory cutoff with
`Qwen/Qwen3.5-9B` tokenizer revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` leaves 8,538 rows and filters
962. The calculation renders the unchanged teacher prompt around every eligible
trajectory and tokenizes it with the official `moonshotai/Kimi-K3` tokenizer at
revision `a590ce090cb049c93a33dfe8c208ec652aa20503`.

Local counts were calibrated against paid Makora responses. On every checked
canary row, Makora prompt usage was exactly the official Kimi raw-prompt token
count plus 21 chat-wrapper tokens. This gives 71,860,759 predicted provider
prompt tokens across the eligible release. Mean prompt length is 8,416.6 Kimi
tokens; p50 is 5,660.5, p90 is 19,434.5, p95 is 22,074.4, p99 is 25,175.6, and
the maximum is 28,472.

## Cost at current Makora rates

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

## Paper-count projection

The paper reports 8,688 training rows across a different five-source mixture.
Those exact trajectories are not materialized here, so their token total cannot
be computed exactly. Applying the measured eligible-release mean of 8,416.6
prompt tokens per row gives 73.12 million prompt tokens and a projected
`$187.57` with ten output tokens per row. Treat this as a planning projection,
not the cost of the paper's actual mix; BashArena and BashBench length
distributions could move it materially. A 10% envelope would be about `$206.33`.
