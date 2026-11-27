# Make paid teacher caching provider-sticky and auditable

Date: 2026-08-30.
Status: current.

- Route synchronous OpenRouter annotation campaigns through a stable
  `session_id` derived from the model and exact prompt prefix, mark that prefix
  with an explicit ephemeral cache breakpoint, and complete one warm-up request
  before concurrent fan-out. This preserves the concatenated prompt text while
  giving provider caches a warm shared prefix.
- For Kimi K3 (`moonshotai/kimi-k3`), prefer Makora for binary-logprob caching;
  it currently has lower input, output, and cache-read rates than Fireworks.
  Pin Makora when endpoint consistency is required, or explicitly order Makora
  before Fireworks when availability is more important.
- Store an exact request-settings fingerprint, actual route metadata, timestamps,
  and provider cache token telemetry in every row. Reject resume when either the
  prompt or request settings drift, and require a canary with nonzero cache reads
  before scaling paid annotation.
- A live synthetic Makora canary verified logprob delivery and reused 8,448 of
  9,627 prompt tokens on its second request. Reported request cost fell from
  $0.024404 to $0.005218; no repository or dataset content was sent.
