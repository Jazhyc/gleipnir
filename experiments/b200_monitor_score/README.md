# Dedicated monitoring score endpoint

Hypothesis: a two-row output projection and vLLM's classification runner can
reduce one-token monitoring latency by removing full-vocabulary projection,
sampling, logprobs, ranks and generated-text handling. Use the selected cached
FP4/Gigatoken/direct-host backbone, causal attention and LAST pooling. Keep KV
and GDN state, chunked prefill, context/token budget and batching unchanged;
prefix caching stays disabled. The classifier copies the merged BF16 head rows
for tokens `0`/`1`, preserves BF16 head output rounding and returns the binary
margin and probability. This is not the cache-free prototype.

`POST /v1/monitor/score` takes `{"model":"monitor","prompt":"..."}`. Supply
the complete rendered monitoring prompt, including the frozen decision suffix;
the endpoint adds no template or special tokens and performs no truncation.
It returns `score`, `margin`, ordered `logits` (`0`, then `1`) and
`prompt_tokens`. Generation options and token-ID inputs are rejected. The
classification server does not expose the generation contract.

Before launch, retire the identity-verified existing server; never keep both on
the GPU. Reuse its staged runtime, merged checkpoint and persistent caches.
Bind source and installed-runtime identities. Check exact classifier row
weights, causal/chunked settings, finite responses and a twenty-row score
canary against the accepted cached serving artifact. Require canary mean score
error <=0.005, correlation >=0.995 and a nonzero adapter effect. Record strict
master agreement separately from inherited finite-quality acceptance.

The initial trial used the same-host cached generation control. Subsequent
comparisons use all three c1 and six c128 repeats of the checksum-bound selected
reference. Measure frozen quick64/c1
and full320/c128 with one excluded warmup, three/six repeats and fresh HTTP
pools per pass. Report input tokens/s, latency bins, score differences, pooled,
per-source and source-macro AUROC, calibration, ties and threshold diagnostics.
Full320 is training-seen systems development, not held-out quality evidence.
Do not promote a new baseline automatically. Stop on provenance/weight drift,
nonfinite output, canary failure, truncation, server failure or suite completion.
No new capacity or final-ID selection. Preserve failures and restore the
selected reference if the candidate cannot provide a usable score service.

```bash
PYTHONPATH=src:. python -m experiments.b200_monitor_score.run --name score01
```

The default entrypoint restores the checksum-bound repaired score recipe from
the live pooling parent, including removing experimental graph settings. It
reads the live parent server before retirement. Run artifacts
live under `results/b200_monitor_score/<name>/`; shared server metadata and logs
remain under `b200_attention_gdn_serving` so the established stop helper can
retire the candidate too. The active server is recorded explicitly as pooling,
not as a reusable generation worker. Changes need matched score/AUROC evidence
because a smaller GEMM can change BF16 rounding.

`score02` is the historical initial endpoint trial. The subsequent validated
mutation repair (`mutation03`) is now the user-selected optimization reference;
see [the decision](../../docs/decisions/b200_monitor_score_reference.md).
The original endpoint [finding](../../docs/findings/b200_monitor_score_endpoint.md)
retains all initial quality and failed-launch evidence.
