# Original-training firewall concept census

Hypothesis: strong learned alignment with the fixed APPS injection direction
is associated with firewall/network-safeguard content, beyond the seven reviewed
extreme harmless/teacher-positive records. Score all 8,688 original training
records using unadapted Qwen3.5-9B and a label-blind concept-presence rubric.
Original trajectory bytes, reasoning and existing middle summaries are retained;
no synthetic injection views, filtering, training or new teacher calls.

The classifier detects discussion, inspection, configuration or manipulation of
firewalls and concrete network-access safeguards. Presence includes benign,
authorized, quoted or resisted cases. Generic networking, API credentials,
ordinary file permissions and process management alone do not count. This is
not a harmfulness judgment. Interpret the scores alongside the preceding manual
review's direct edits and weaker summary evidence; presence does not imply edits.

Use continuous normalized next-token 0/1 logprobs with thinking off, temperature
zero and a one-token completion. Pin the existing 9B checkpoint revision and
all prompt/input/source identities; preserve raw decision logprobs, full-context
token counts and timestamps. Stock BF16 eager vLLM 0.31 is an explicit backend
exception: the custom optimized envelope is validated for 4B geometry, not 9B.
Use the existing B200 sequentially after retiring the recorded Lens worker;
preserve the merged checkpoint and shared compiler caches. No capacity lifecycle.

Reuse `gleipnir.evaluation.concept_census`, extracted from the injection census's
rendering, binary-logit, resumption and execution helpers, with explicitly
configured controls and concept name. Its old entrypoint remains compatible.
Historical injection receipts keep their executed sources;
do not resume them against changed live source hashes.

Freeze thresholds 0.1/0.5/0.9 and top 1/5/10% activation-change bands before
scoring. Fixed synthetic presence/absence controls diagnose the rubric; report
misses without tuning it on outputs. Stop on hash/membership/token drift,
truncation, missing/nonfinite logprobs, OOM/backend failure or complete coverage.
Completion requires 8,688 unique finite scores and local collection checks.

Join to the completed original-data activation census by ID and trajectory hash.
Report firewall score/log-odds versus raw and cosine alignment change, pooled
and within source/behavior-label/teacher-target groups; distributions, ties,
fixed-band intersections and high-change low-firewall counterexamples. Source,
teacher targets and length can confound pooled patterns. Keep original duplicate
exposure weights and report a unique-trajectory sensitivity view. No verified
concept truth or calibrated prevalence, and no causal data attribution claim.
Leave source behavior labels separate from classifier labels. APPS/ID quality
evaluation, filtering and retraining are outside this audit.

Run `python -m experiments.training_firewall_census.run`, then
`python -m experiments.training_firewall_census.analyze`.
Configuration: `config.json`; rubric: `prompt.txt`.
Artifacts: `results/training_firewall_census/qwen35_9b_v1/`.
Startup/progress monitoring is active-turn only; no heartbeat tool is available.

Full-length anchor misses require interpreting low scores as unverified absence,
even when short controls pass. Preserve misses without revising the frozen rubric.
The analysis adds a disclosed post-hoc literal firewall-keyword cross-check and
manual counterexample review; record its motivation and results in the findings.
