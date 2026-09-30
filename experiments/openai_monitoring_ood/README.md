# OpenAI full-teacher OOD monitoring benchmark

Evaluate direct OpenAI GPT-6 Luna on all 6,395 rows of the frozen six-source OOD
suite with the unchanged full teacher prompt. This is inference benchmarking,
not student training or distillation. Compare with the existing Kimi K3,
Kimi K2.6, and Qwen3.5-27B full-prompt binary-logprob baselines.

Status: completed on all 6,395 rows and audited on 2026-09-30. Mean-OOD
pAUROC@20 is **0.836544**, and Mean-OOD AUROC is **0.928957**. Ordinary-input
comparison cost is **$0.836167/1K**; usage with reported cache reads/writes is
**$1.038409/1K**, including recovery attempts. See the
[finding](../../docs/findings/openai_luna_ood_benchmark.md) for per-source
results, SHADE-Arena's high false-positive rate, and complete provenance.

The frozen hypothesis, intervention, baselines, selection rule, request settings,
and stop conditions are in `config.json`. Use Standard processing
(`service_tier=default`) and 40 concurrent workers. Reasoning is disabled,
temperature is 1, and `top_p=1` avoids the cutoff found in the synthetic
[compatibility canary](../../docs/findings/openai_luna_logprobs_canary.md).

The full teacher prompt is the sole user message, unchanged. The generated
completion must be exactly `Prediction:0` or `Prediction:1`, allowing surrounding
whitespace. Scores normalize the two literal terminal digit logprobs. Both
must be returned and finite; missing tokens are failures, never inferred from
the generated label. Preserve raw small positive numerical roundoff up to
`1e-4`, use the returned values without clamping, and report its frequency.

Before scaling, check 12 non-OOD rows selected solely by SHA256 of stable IDs,
then the longest OOD prompt by character length. These gates inspect only
format, identity, numeric validity, reasoning usage, and coverage. They do not
select based on model quality. The longest-prompt score is reused in the full
suite. No OOD result changes the prompt, threshold, or selection rule.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.openai_monitoring_ood.run \
  --tokens-per-minute 1800000 --coverage-repeats 3 --incomplete-repeats 3
```

The first full pass saved 319 valid OOD rows before exhausting transient retries
against the account's 2M tokens/minute limit. Resume with 40 worker slots and
paced starts, counting 105% of the stored Kimi tokens plus a 48-token allowance:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.openai_monitoring_ood.run \
  --tokens-per-minute 1800000
```

At 1,461 saved rows, one AgentDojo response omitted the opposite literal label
from top20 even with `top_p=1`. The run stopped and preserved that failure.
A separately frozen, bounded identical-request check recovered both scores on
its first repeat. Resume with at most three identical coverage repeats:

```bash
PYTHONPATH=src .venv/bin/python -m experiments.openai_monitoring_ood.run \
  --tokens-per-minute 1800000 --coverage-repeats 3
```

This operational amendment accepts the first response satisfying the original
format and numeric coverage contract. It never inspects a label or performance
metric, changes a prompt or request field, or imputes a missing probability.
Preserve and count every failed response and recovery. Numerical nondeterminism
between identical calls means recovered distributions can differ; report this
limitation. The original seven-attempt HTTP ceiling also bounds coverage repeats.

Two SHADE-Arena responses later reached the original 16-token cap after adding
extra output to a prediction. The first recovered on its first identical repeat.
The final resumption also permits at most three unchanged repeats of explicitly
incomplete `max_output_tokens` responses, only with matching model/settings and
zero reasoning tokens. Do not increase the output cap or extract scores from a
partial response. Every accepted response must satisfy the original contract.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.openai_monitoring_ood.run \
  --tokens-per-minute 1800000 --coverage-repeats 3 --incomplete-repeats 3
```

Pacing and source hashes are recorded in `execution_events.jsonl`. These are
operational resumption settings; request semantics and cache identities remain
frozen. Retry waits honor at least Retry-After and exponential backoff.

`OPENAI_API_KEY` is loaded from the ignored `.env`; never logged. Every HTTP
attempt, including parse failures and transient retries, is appended to ignored
`attempts.jsonl`. Valid scores are flushed individually to `predictions.jsonl`.
Frozen input, manifest, config, settings and rendered-prompt hashes protect
resumption. A final complete-ID audit precedes metric calculation. Report macro,
pooled, weighted and per-source ranking; Brier score, fixed 0.5 threshold,
calibration bins, score ties, and usage-derived cost. Bootstrap intervals are
source-label-stratified row bootstraps, not lineage confidence intervals.

Budget USD 10 including canaries and retries. Reserve each outstanding request
at 130% of its stored Kimi token count (UTF-8 byte count when absent on a canary)
and the maximum output cap, then charge
all returned input tokens at the higher cache-write rate for budget enforcement.
Transport failures retain the estimate because billing is unknown. Final costs
separate an uncached comparison coordinate from pricing using reported cache
reads and writes, plus cache-write sensitivity; API usage is not a billing receipt.

Logs: `logs/local/openai_monitoring_ood/run_v1.log`.
Artifacts: `results/openai_luna_ood_benchmark/run_v1/`.
This session has no scheduled agent-heartbeat tool. Active-turn checks cover
startup and progress but cannot promise follow-ups after the turn ends.

## GPT-5.6 Luna matched comparison

The user authorized a full matched benchmark and increased concurrency on
2026-09-30. `config_56_luna.json` freezes the hypothesis that the older Luna
may score monitoring examples better under the same zero-reasoning interface.
Only the requested model changes; the prompt, rows, score normalization,
threshold, metrics, canary selection, and previously recorded bounded recovery
policy remain identical. Compare against completed GPT-6 Luna, Qwen27B,
and Kimi baselines without choosing settings on OOD results.

Use 80 workers, twice the previous concurrency. Before launching the full suite,
read the GPT-5.6 Luna canary's actual rate-limit headers. Initial pacing is
1.9M estimated tokens/minute, at most 95% of the reported model-specific TPM.
Record any operational pacing adjustment before resuming unchanged requests.
The previous GPT-6 Luna run had 264 HTTP 429 errors at a 2M TPM limit;
increasing worker count does not remove that quota.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.openai_monitoring_ood.run \
  --config experiments/openai_monitoring_ood/config_56_luna.json \
  --tokens-per-minute 1900000 --coverage-repeats 3 --incomplete-repeats 3 \
  --stop-after-canary
PYTHONPATH=src .venv/bin/python -m experiments.openai_monitoring_ood.run \
  --config experiments/openai_monitoring_ood/config_56_luna.json \
  --tokens-per-minute 1900000 --coverage-repeats 3 --incomplete-repeats 3
```

Standard short-context prices checked on 2026-09-30 are $0.20/M ordinary input,
$0.02/M cached input, $0.25/M cache writes, and $1.20/M output. Budget enforcement
and final accounting now use the frozen configuration's prices; the original
GPT-6 Luna defaults remain for compatibility with its audit. Reusing the
previous run's token volumes projects $10.70 at ordinary rates or $13.29 with
its observed cache writes and reads. The conservative campaign stop bound is
$20 including canaries and all recovery attempts. Actual token volumes can
differ between models.

Logs: `logs/local/openai_monitoring_ood/run_56_luna_v1.log`.
Artifacts: `results/openai_56_luna_ood_benchmark/run_v1/`.
Preserve raw failed requests and audit complete IDs, settings, prompts, scores,
usage and artifact hashes before reporting final results. This session has no
scheduled agent-heartbeat tool; monitoring uses active-turn progress checks.

The GPT-5.6 run stopped at 2,629 saved rows after a completed response appended
extra characters to `Prediction:1`. Before resumption, record a bounded
amendment allowing at most three identical retries for completed malformed
prediction text, only with matching model/settings and zero reasoning tokens.
This does not accept partial or malformed outputs, alter the request, or choose
by label or quality. Retain every attempt and the first fully valid response.
The original seven-attempt HTTP ceiling remains. The initial invalid response
is preserved in `attempts.jsonl`; the amendment is in
`completed_format_recovery_plan.json` and `execution_events.jsonl`.

```bash
PYTHONPATH=src .venv/bin/python -m experiments.openai_monitoring_ood.run \
  --config experiments/openai_monitoring_ood/config_56_luna.json \
  --tokens-per-minute 1900000 --coverage-repeats 3 --incomplete-repeats 3 \
  --format-repeats 3
```
