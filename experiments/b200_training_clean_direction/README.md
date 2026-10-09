# Complete original-training direction census

User correction: evaluate **every original clean training row**, with no added
synthetic injection views. Hypothesis: the fixed APPS unit u20's learned
alignment is concentrated in particular original sources, behavior/teacher
groups or naturally occurring injection-like text. This is activation-based
candidate discovery, not causal training-example attribution.

Freeze all 8,688 rows from the original regular `student_rows.jsonl`, exactly
once per model, with original source bytes/lineage/labels/teacher targets.
The current BF16/SDPA augmented-trained adapter remains the trained model under
study; the correction changes inputs, not its checkpoint. No augmentation
artifact is loaded into the population. Original traces can themselves contain
natural injection attempts, quoted tests or reasoning. Do not clean or truncate
those historical inputs. Render the exact training chat/prefix with thinking
disabled; targets and census labels stay outside model-visible prompts. Native
counts use the effective HF tokenizer, not its raw JSON alone.

Compare base and trained final-token post-layer residuals at 20/31 along the
unchanged trained-model APPS u20/sign. Primary measures: layer-20 projection,
cosine, trained-minus-base change and norm; 31 is ancillary. Both models share
this coordinate axis without feature-alignment fitting. Fresh complete passes
keep the same order/batches, rather than mixing the previous small sample's
scores/activations. Existing sample results are exploratory prior evidence.

Join pinned existing Qwen3.5-9B injection-presence census by unique original ID
and exact trajectory hash, and clean-source soft teacher targets by their
original prompt lineage. The census is a model flag with documented false
positives/misses, not injection ground truth. No paid teacher calls or new
semantic classifier. Report pooled/source/dataset/label/teacher/census groups,
length/norm controls, teacher and census associations within source/label,
and how much positive projection/change is carried by each source and the top
1/5/10 percent ranked rows. These are shares of measured scalar mass, not shares
of parameter updates or causal influence. Signed means can cancel.

Archive complete rankings and top/bottom 100 bounded excerpts for trained
alignment, raw change, cosine change and base alignment. Include teacher-hard
disagreement groups, source/label mixtures, score ties and candidate-review
flags at fixed top 1/5/10-percent rank bands and inherited census >=.5/.9.
All bands are descriptive, not selected filters. Never remove a harmful example
merely because its alignment is large; preserve the original dataset. Actual
filtering/retraining is a subsequent user decision requiring frozen data and
ID/APPS guards. Earlier broad census removal failed to resolve APPS alarms.

Reuse the existing sequential orchestration in `gleipnir.evaluation.direction_campaign`,
with this config's isolated artifact/log roots. Use identical BF16 eager Lens
research baseline, c16/batch32, prefix-off, native tokenizer, scheduler and
backend for both models. Verify base/merged/source/runtime identities and the
existing master/eager canaries (MAE <=.020/correlation >=.99; eager <=.005/.995;
nonzero adapter effect), native BF16 dispatch and capture coverage/finiteness.
Retain the cached/full-head logit drift limit .25. Stop on a failed gate,
identity drift, missing rows, context overflow, nonfinite values, OOM or complete
coverage. Preserve failures and batch hashes. Restore/gate the trained engine
after base capture. No training, filtering, dependencies or capacity changes;
weights remain remote and shared caches remain intact.

Before launch, freeze config, sources, workload/input hashes, effective tokenizer
and current resident receipt. Collect lossless raw BF16 captures in FP32 and
independently check all original joins, counts, dots/deltas and rankings. Report
input tokens/s, requests/s and queued p50/p95 latency as one-pass capture costs.
Monitoring is active-turn only; no heartbeat tool is available.

Run `python -m experiments.b200_training_clean_direction.prepare`, then
`python -m gleipnir.evaluation.direction_campaign --config
experiments/b200_training_clean_direction/config.json` in the pinned runtime.
Artifacts: `results/b200_training_clean_direction/clean_direction01/`.
