# Direct OpenAI GPT-6 Luna logprob compatibility canary

Date: 2026-09-30 (Europe/Amsterdam; requests on 2026-09-29 UTC).
Status: completed synthetic API compatibility check, not a monitoring benchmark.

## Question and design

Does direct OpenAI `gpt-6-luna` return actual token logprobs with reasoning
disabled, and can the interface supply both literal `0` and `1` scores for
binary teacher targets? The earlier access check failed with
`401 expired_secret_key`; the user refreshed the local `.env` credential before
this canary. Credentials were loaded without printing or persisting them.

The frozen initial manifest specifies three short, original synthetic monitoring
examples: accurate reporting of failed tests, intentional concealment of failed
tests, and a test-success claim without corroborating evidence. The instruction
asks for one digit, `0` for benign or `1` for problematic. No dataset, held-out
evaluation, or repository content was supplied. There is no model promotion or
prompt-selection decision.

The initial five-request limit covers three Responses requests with reasoning
`none`, one Chat Completions request with reasoning `none`, and a Responses
negative control with reasoning `low`. An adaptive sixth request corrected the
Chat Completions limit error and used the same uncertain input as Responses;
its request and identity were recorded before sending it. No retries were made.

## Observations

| Request | HTTP | Output | Literal logprobs returned |
| --- | ---: | --- | --- |
| Responses, benign, `none`, top 20 | 200 | `0` | `0: -0.000050`; `1` absent |
| Responses, concealment, `none`, top 20 | 200 | `1` | `1: +0.000004`; `0` absent |
| Responses, uncertain, `none`, top 20 | 200 | `1` | `1: -0.059917`, `0: -2.844639` |
| Chat Completions, benign, `none`, top 20 | 400 | — | `top_logprobs` must be at most 5 |
| Responses, benign, `low`, top 20 | 400 | — | Logprobs unsupported with reasoning enabled |
| Chat Completions, uncertain, `none`, top 5 | 200 | `1` | `1: -0.060921`, `0: -2.828480` |

All four successful requests returned actual token-level logprob objects and
reported zero reasoning tokens. The Responses uncertain example gives
`exp(lp1) / (exp(lp0) + exp(lp1)) = 0.941845`; the corresponding Chat
Completions score is approximately `0.9409`. These are normalized probabilities
over the two returned literal tokens, not validated calibration measurements.

Responses accepted `top_logprobs=20` but returned only one alternative on each
clear example and two on the uncertain example. Requesting more alternatives
therefore did not guarantee delivery of both labels. Preserve the small positive
returned logprob on the concealment example as raw evidence; mathematical log
probabilities cannot exceed zero, so a future cache must define its numerical
validation policy rather than silently treating every returned float as valid.

The successful requests processed 268 input and 19 output tokens, with zero
cache reads or writes. At published short-context Standard rates, their
estimated token cost is `$0.0000363`. This is a rate-based estimate, not a billing
receipt. Single-request wall times ranged from 0.74 to 2.19 seconds; this tiny
sequential canary does not establish production latency or throughput.

## Updated belief and limits

Direct GPT-6 Luna supports logprobs on both Responses and Chat Completions with
reasoning effort `none`; a live reasoning-`low` request rejected logprobs.
This makes Luna a possible direct-logprob teacher candidate, subject to a
representative format and coverage canary. The observed missing counter-label
scores prevent treating it as a drop-in source of complete binary distributions.
Keep incomplete rows marked as parse/coverage failures; do not reconstruct a
missing label as one minus the chosen token probability or substitute a hard
decision or elicited confidence.

The ambiguous example was assigned a high problematic score despite limited
evidence. This illustrates why compatibility alone supplies no evidence of
monitoring quality, calibration, or transfer.

The separately announced Decisions API is in limited preview, with broader
release planned in the coming days. Its announcement supplies no probability
contract. These observations apply to existing direct inference endpoints and
do not establish Decisions API behavior or account access.

## Artifacts and sources

Ignored local artifacts:

- `results/openai_decision_api_probe/models.json`: earlier expired-key response.
- `results/openai_decision_api_probe/20260929T230728Z/manifest.json`: frozen
  synthetic prompts, exact initial requests, request hashes, and stop rule.
- `results/openai_decision_api_probe/20260929T230728Z/responses.jsonl`: all five
  initial outcomes, raw logprobs, model IDs, usage, timestamps, and request IDs.
  SHA-256: `bdd9a71b5f8d30e76f7984adefcf6841b3fb77f7ecfc50736884a3469c8b455a`.
- `results/openai_decision_api_probe/20260929T230728Z/chat_followup_request.json`
  and `chat_followup_response.json`: recorded adaptive request and raw response.

Official sources checked during the investigation:

- [GPT-6 API parameter guidance](https://developers.openai.com/api/docs/guides/latest-model#migration-quickstart).
- [GPT-6 Luna model](https://developers.openai.com/api/docs/models/gpt-6-luna).
- [Published pricing](https://developers.openai.com/api/docs/pricing).
- [DevDay Decisions API announcement](https://openai.com/index/devday-2026-recap/).
