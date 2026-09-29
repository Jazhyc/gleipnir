# Direct OpenAI GPT-6 Luna logprob compatibility canary

Date: 2026-09-30 (Europe/Amsterdam; requests on 2026-09-29 UTC).
Status: completed synthetic API compatibility check, not a monitoring benchmark.
Follow-up corrected the initial missing-score diagnosis: explicit `top_p=1`
restored both label scores on all three Responses examples. The earlier requests
had echoed an effective default `top_p=0.98`.

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
The initial conclusion that Luna could not supply complete binary targets was
premature: the initial probe did not disable the effective nucleus cutoff.
Explicit `top_p=1` resolves missing label scores on the tested Responses inputs,
making Luna a possible direct-logprob teacher candidate. A representative format
and coverage canary is still required: a top-k interface does not guarantee named
token scores on every possible prompt. Keep incomplete rows marked as
parse/coverage failures; do not reconstruct a missing label as one minus the
chosen token probability or substitute a hard decision or elicited confidence.

## Follow-up: isolate the effective nucleus cutoff

Inspection of the raw initial Responses objects revealed `temperature=1.0` and
`top_p=0.98`; both parameters had been omitted from the requests. The one-token
and two-token alternative lists were consistent with nucleus truncation. The
follow-up froze a three-request manifest before execution and repeated exactly
the original instructions, inputs, and settings, adding only explicit `top_p=1`.
The effective temperature remained `1.0`. The stop rule was three requests or
the first HTTP/transport error.

| Example | Alternatives returned | `logprob(0)` | `logprob(1)` | Normalized `P(1)` |
| --- | ---: | ---: | ---: | ---: |
| benign | 19 | -0.000175 | -8.644638 | 0.000176068 |
| concealment | 18 | -10.198551 | -0.000034 | 0.999962776 |
| uncertain | 19 | -3.376644 | -0.034763 | 0.965837960 |

Every request succeeded, echoed `top_p=1.0`, and returned both literal scores.
The two labels occupied ranks 1 and 2 in every returned list: the missing label
is therefore not intrinsically outside the top 20 on these inputs. The observed
change strongly supports the effective nucleus cutoff as the cause of the
original missing alternatives. This is an inference from matched API requests,
not a claim about undocumented server internals.

The uncertain score changed between calls, so this follow-up establishes
coverage, not exact numerical parity across settings or repetitions. These
three requests processed 201 input and 15 output tokens, with zero reasoning or
cache tokens. Both-label coverage at explicit `top_p=1` has been verified on
Responses; the Chat Completions follow-up has not been repeated at that setting.

For a Responses teacher canary, use `reasoning={"effort": "none"}`,
`temperature=1`, `top_p=1`, `include=["message.output_text.logprobs"]`, and
`top_logprobs=20`, and validate literal-score coverage and numeric validity on
every row. Do not assume backend defaults provide the same score distribution
as an explicit untruncated request.

The ambiguous example was assigned a high problematic score despite limited
evidence. This illustrates why compatibility alone supplies no evidence of
monitoring quality, calibration, or transfer.

The separately announced Decisions API is in limited preview, with broader
release planned in the coming days. Its announcement supplies no probability
contract. These observations apply to existing direct inference endpoints and
do not establish Decisions API behavior or account access.

## GPT-6.1 Sol and GPT-6 Astra: logprobs unavailable

A separate four-request synthetic Responses canary checked `gpt-6.1-sol` and
`gpt-6-astra`. Its manifest was frozen before execution. For each model, one
request used allowed reasoning effort `low` plus the logprob include field and
top-five alternatives; another used reasoning effort `none` without requesting
logprobs, isolating the reasoning-setting restriction. All requests used the
same original short instruction to return `0`, capped output at 32 tokens, and
omitted `temperature` and `top_p`. There were no retries or dataset inputs.

Both models returned HTTP 400 for both conditions:

- `low` plus logprobs: `unsupported_parameter` on `include`, with message
  `logprobs are not supported with reasoning models.`
- `none` without logprobs: `unsupported_value` on `reasoning.effort`, with
  allowed efforts `low`, `medium`, `high`, `xhigh`, and `max`.

These live results match the official model and migration guidance: mandatory
reasoning prevents using the `none` mode that enables Luna's token logprobs.
Neither tested model currently supports our direct binary-logprob teacher
interface through the standard API. Asking for a generated probability would
constitute a different supervision method and would not supply the underlying
two-token distribution. All four error responses had no usage object; no
successful inference was performed in this compatibility canary.

## GPT-6 Sol: both binary scores available with reasoning disabled

A subsequent three-request Responses canary repeated the original synthetic
inputs with `model="gpt-6-sol"`, `reasoning={"effort": "none"}`,
`temperature=1`, `top_p=1`, `top_logprobs=20`, and the logprob include field.
The manifest and a three-request stop rule were recorded before sending calls.
No prompts were changed or selected based on outcomes.

All three requests returned HTTP 200, echoed the requested model/settings,
reported zero reasoning tokens, and supplied 20 alternative token scores
including both literal `0` and `1`:

| Example | `logprob(0)` | `logprob(1)` | Normalized `P(1)` |
| --- | ---: | ---: | ---: |
| benign | +0.000008 | -14.927895 | 0.000000329 |
| concealment | -13.363712 | -0.000015 | 0.999998429 |
| uncertain | -0.000973 | -6.939995 | 0.000968279 |

The tiny positive raw logprob on the benign example is preserved, as with the
initial Luna numerical anomaly; require a recorded numerical tolerance policy
before accepting such values in a scaled cache. The normalized scores in this
table describe the returned pair and do not establish calibration. Sol's
different decision on the uncertain input is descriptive, not evidence of
comparative monitoring quality from this synthetic compatibility check.

This verifies the binary score interface on GPT-6 Sol, unlike GPT-6.1 Sol's
mandatory-reasoning interface. Three synthetic inputs do not establish complete
coverage on actual trajectories. The successful calls processed 201 input and
15 output tokens, with no cache reads or writes.

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
- `results/openai_decision_api_probe/top_p_20260929T231345Z/manifest.json` and
  `responses.jsonl`: frozen matched `top_p=1` follow-up and raw responses.
  Response SHA-256:
  `cba64e102c0b1260510038ba0add1d2bfbb94b72a1140ac5ad1fd415cd705f70`.
- `results/openai_decision_api_probe/sol_astra_20260929T231614Z/manifest.json`
  and `responses.jsonl`: frozen Sol/Astra restriction check and all four raw
  HTTP-400 responses. Response SHA-256:
  `8a6637c6b33478d8a6490ebc1743943b0742f6e5e9b12fa400b2aa5772a92c32`.
- `results/openai_decision_api_probe/sol6_20260929T231728Z/manifest.json` and
  `responses.jsonl`: frozen GPT-6 Sol compatibility check and three raw
  successful responses. Response SHA-256:
  `6f8583b3167bcedd7b175566085e9ddd8a00f2fffa2a5e73a30bfb3ea4b00b3b`.

Official sources checked during the investigation:

- [GPT-6 API parameter guidance](https://developers.openai.com/api/docs/guides/latest-model#migration-quickstart).
- [GPT-6 Luna model](https://developers.openai.com/api/docs/models/gpt-6-luna).
- [GPT-6 Sol model](https://developers.openai.com/api/docs/models/gpt-6-sol).
- [GPT-6.1 Sol model](https://developers.openai.com/api/docs/models/gpt-6.1-sol).
- [GPT-6 Astra model](https://developers.openai.com/api/docs/models/gpt-6-astra).
- [Published pricing](https://developers.openai.com/api/docs/pricing).
- [DevDay Decisions API announcement](https://openai.com/index/devday-2026-recap/).
