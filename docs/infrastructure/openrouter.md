# OpenRouter

`gleipnir-openrouter` reads prompt records from JSONL and checkpoints binary
label logprobs to another JSONL file. Each input row needs `id` and `prompt`;
arbitrary other fields are copied into `metadata` when explicitly supplied by
the caller. Example:

```bash
gleipnir-openrouter \
  --input data/teacher_prompts.jsonl \
  --output results/deception_distillation/teacher/openrouter.jsonl \
  --model moonshotai/kimi-k3 \
  --provider-only makora \
  --no-allow-fallbacks \
  --concurrency 16
```

The command requires `OPENROUTER_API_KEY`, requests literal terminal `0|1`
logprobs, stores prompt and request-setting hashes, and resumes only when both
identities match. Kimi K3 is currently available as `moonshotai/kimi-k3`; Makora
is the lowest-priced OpenRouter endpoint with logprob and prompt-cache support.
Keep `--provider-only makora` when endpoint consistency matters. For a more
availability-oriented campaign, use `--provider-order makora --provider-order
fireworks` and leave fallbacks enabled. These are provider tags, not display
names, and are deliberately lowercase.

Provider caching uses an explicit ephemeral `cache_control` breakpoint on the
exact prefix shared by every input prompt. The CLI also hashes that prefix into
a stable OpenRouter `session_id`, completes one request synchronously to warm it,
then fans out at the requested concurrency. OpenRouter can consequently keep the
campaign on the endpoint holding the prefix cache. Use `--session-id` to share a
deliberate cache session across separately materialized shards, or
`--no-explicit-cache`, `--no-sticky-routing`, and `--no-warm-cache` to disable
the corresponding behavior. Keep the long, invariant teacher instructions at
the beginning of every prompt; only the exact shared prefix is reusable.

Each result row records the requested settings and their SHA-256 identity, the
actual model/provider, router metadata, raw usage, timestamps, and normalized
`cache_usage.cached_tokens`, `cache_write_tokens`, and `cache_discount`. The CLI
prints aggregate cache reads/writes and the cached fraction of input tokens at
shutdown. A scale-up should first run a small canary and confirm nonzero cached
tokens after the warm-up row.

On 2026-08-30, a two-request synthetic canary pinned to `makora` confirmed both
`logprobs`/`top_logprobs` and explicit prefix caching. The warm request reported
0 cached tokens; the second request reported 8,448 cached tokens out of 9,627
input tokens. Reported request cost fell from $0.024404 to $0.005218. Current
OpenRouter catalog rates were $2.55/M input, $12.75/M output, and $0.256/M cache
read for Makora, versus $3.00/M, $15.00/M, and $0.30/M for base Fireworks.
