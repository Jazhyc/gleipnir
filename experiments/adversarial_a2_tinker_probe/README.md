# Tinker connectivity probe

Frozen before sampling. One synthetic sampling call, no APPS data and no monitor
scores, to verify the user's existing Tinker credentials and GLM-5.3 access.
Model: `zai-org/GLM-5.3:peft:262144`. Prompt: `The sum of 2 and 2 is`.
One sample, temperature 0, at most 32 generated tokens, no application retries.
This is a raw-completion transport test, not a chat-quality or reliability study.
Use the SDK tokenizer for model input. No training clients, weights or checkpoints.
Stop if encoded input exceeds 100 tokens. Bound cost below $0.01 using published
$3.78/M prefill and $12.15/M sampling prices on 20 October 2026:
https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/
Keep SDK/model identity, timing and returned token count in ignored results.
Read the key from the wrapper `.env` without copying or logging it. Do not merge
this provider's outputs into the active OpenRouter high-effort A2 run.

## Result

Authentication succeeded; the account exposed 38 models, including full GLM-5.3
and GLM-5.3-Flash. A GLM-5.3 sampling call succeeded using SDK 0.34.0. It consumed
9 prompt tokens and returned 32 tokens, beginning with the correct continuation
`4.`. Sampling took 2.95 seconds after client/tokenizer setup and stopped at the
configured token limit. Estimated uncached token cost: $0.000423. Receipt:
`results/adversarial_a2_tinker_probe/probe.json`, ignored. The missing-PyTorch
warning is expected: this machine only tokenizes; inference occurs on Tinker.
This single tiny raw-completion request establishes access, not sustained
throughput, A2 edit quality, high-effort parity or lower error rates. The existing
OpenRouter high-effort run remains separate and active. A Tinker A2 run needs a
separate protocol, model-correct chat rendering and a sampling budget reflecting
Tinker's higher token prices.
