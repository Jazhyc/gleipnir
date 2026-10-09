# JudgeDeceiver signal versus decision transfer

User-authorized follow-up to `b200_projection_judge`. Test whether the unchanged
APPS-fitted unit layer-20 direction responds to JudgeDeceiver injections even
though its full projection changes preference accuracy little. No direction,
threshold, layer, strength, classifier or checkpoint fitting/selection.

Reuse all 4,188 original rendered A/B variants, six held-out query groups,
252 pairs and both orders. Pair each of 3,684 injections with its clean parent
by pair/order; preserve sources, conditions and labels. Original provenance,
construction-label and unresolved-license limitations remain defined in
`experiments/judge_injection_continuation/README.md`. The previous evaluation
already used this test cohort; this is exploratory mechanistic follow-up,
not an untouched confirmatory test or an independent training-seed replicate.

Primary readout: final-token post-layer-20 complete residual dot the original
APPS unit u20, sign fixed. Capture layers 20 and 31, the latter required for
existing exact fused-normalization A/B readout and an ancillary removal check.
Do not choose another layer after observing results. Score original unedited
and full-projection prompts; beta 1/zero centers at all 32 layers/all tokens
exactly matches the prior arm. Report paired injection-minus-clean projection
shifts, sign/tie frequencies, matched-clean-weighted injection-detection AUROC,
and source/query breakdowns. AUC negatives repeat each injection's own parent,
so group and duplication dependence must be disclosed. Direction separability
is encoded information, not proof of semantic recognition or causal use.

Make two fixed controls per injection by replacing its single contiguous
insertion with deterministic neutral prose or whitespace from config. Keep
the same insertion location and entire clean prompt otherwise intact. Match
the attacked complete prompt's native tokenizer count within one token using
only text/token counts, never activations, labels or model outcomes. Preserve
attack/clean parent IDs, inserted-text checksums, offsets, target/actual counts
and duplicate prompt identities. Neutral prose contains no judge directives;
whitespace preserves lexical candidate content. Added prose can still affect
answer quality, so copied preferences are bookkeeping, not asserted ground
truth for a new control accuracy benchmark. Compare projection shifts for
attack versus each control with matched weighting and by source/query.

Relate original injection shifts to label-adjusted correct-answer A/B margin
changes and to full-projection decision changes. Use descriptive Spearman
coefficients without independent-row p-values. Report all six query groups
and source/query macro summaries; thousands of dependent suffix/order variants
are not thousands of independent examples. No direction reorientation,
benign-control selection, layer sweep or seed-variance interpretation.

Reuse the already passing native/master/merge/readout receipts and current
SDPA BF16 eager Lens engine without a restart. Bind its resident command and
sources. Require capture shape/positions/finiteness, native token identity,
canary reproduction against saved A/B scores (MAE <=.005/correlation >=.995),
and exact beta-zero/next-plain restoration. Require projected absolute u20/u31
components <=2% of residual norm, a conservative predefined BF16 allowance;
report actual absolute/relative residuals and cohort attenuation, not just pass.
This is not a new whole-backbone FP32-master A/B reference. Reuse prior complete
score files as numerical controls; activation capture necessarily makes fresh
requests, whose A/B differences are disclosed.

Freeze prepared workloads, control construction receipts, sources/config and
resident identity before capture. Resumable 128-row batches at concurrency 32
save raw native-BF16 values losslessly in finite FP32 arrays, score/projection
metadata, timing and tensor checksums. Stop on input/source/resident drift,
unmatched insertions, token-count mismatch, failed gates, nonfinite values,
missing capture/state cleanup, OOM or completion; preserve failed receipts.
Collect results/activations/audits and report input tokens/s, requests/s and
latency as one-pass capture costs. Keep the passing B200 engine warm, merged
weights remote and shared caches unchanged. No new capacity or dependencies.
Monitoring is active-turn only; no heartbeat tool is available.

Run `python -m experiments.b200_judge_direction_signal.prepare`, then
`python -m experiments.b200_judge_direction_signal.run` in the pinned serving
runtime. Artifacts use `results/b200_judge_direction_signal/`.
