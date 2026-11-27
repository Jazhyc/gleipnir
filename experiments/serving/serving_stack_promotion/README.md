# Serving stack promotion check

Status: completed 2026-11-27; protocol frozen before the B200 run.

**Question.** Does the optimized serving runtime, promoted from its
experiments into `gleipnir.serving.stack` (29 files moved, imports and
checkout lookups rewritten, no arithmetic change), serve the released
Gleipnir 4B and 27B exactly as before?

**Intervention.** This branch: recorded commands resolve onto the promoted
modules through `gleipnir.serving.sources.promote_command` and
`stack/promotion.json` (`status: candidate`). **Baseline:** `main` at
`1bff5636`, the code the accepted recipes were last run from. Both run on the
same Pod from `ghcr.io/jazhyc/gleipnir-serve:1.2` (its runtimes, kernels and
receipts), each from a clean code tree with the receipt-pinned
`src/gleipnir/__init__.py` (`b6acfaad…`) restored, as in the image.

**Measurement** (user decision, 2026-11-27: the question is determinism, so
the synthetic identity check suffices; no ID pass or throughput timing). Per
model, in the image's default mode (larger 4B graphs, Gigatoken), through the
image entrypoint (`fetch`, `serve`): `image/identity.py` scores 248 synthetic
prompts (330 to 7,170 tokens, covering every graph size) one at a time, and
the branch's scores are compared with `main`'s bit for bit.

**Acceptance rule (frozen).** Accept the promotion when every identity score
of both models is identical to `main`'s. Then set `promotion.json` to
`accepted` (user decision). **Stop conditions:** any startup failure other
than the known MXFP8 compile race, or any identity difference; report and stop
without retrying variants.

Outputs: `results/serving_stack_promotion/{main,branch}/`, logs in
`logs/runpod/serving_stack_promotion/`.

## Results (2026-11-27)

US-NE-1 B200 Pod `x4tbfzde1f4z88` ($8.99/hour, Runpod's Secure Cloud list
price that day; `GPU-687c5627…`, Intel Xeon 6960P, driver 595.112.07), from
`gleipnir-serve:1.2`, no volume.

| Model | `main` ready | Branch ready | Rows | Rows changed (score, logits, prompt tokens) |
|---|---:|---:|---:|---:|
| 4B | 140 s | 75 s | 248 | 0 |
| 27B | 316 s | 115 s | 248 | 0 |

Every score of both models is identical to `main`'s, so the acceptance rule
holds. `main`'s start times include the checkpoint download and a cold compile;
the branch's reuse the compile caches of its earlier attempts.

The first two branch attempts stopped at startup checks, before any model
loaded, on three sources the promotion had edited although receipts bind
their bytes. Each had only a renamed import, so each was restored to `main`'s
bytes (old names resolve through the aliases):

- `experiments/b200_context_million/native.py` and
  `experiments/serving/qwen38_27b_serving/mxfp8_envelope.py`, bound by the 4B
  1M and 27B MXFP8 long-context receipts (`extended envelope source changed`);
- `src/gleipnir/serving/frost_wrappers.py`, which checks its own hash against
  the host-wrapper receipt (`host wrapper validation/source drift`).

`tests/serving/test_stack_promotion.py` now fails on any receipt reachable from
the release stack whose recorded sources stop matching, beyond those that
already differ on `main`. Rows, logs and the failed attempts are in
`results/serving_stack_promotion/` and `logs/runpod/serving_stack_promotion/`
(18 files, SHA-256 verified after transfer).
