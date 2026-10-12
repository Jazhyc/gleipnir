# Restoration preparation: original boundaries still required

The 2026-10-12 local preparation covers all 6,395 frozen IDs and unchanged hard
labels. It carries forward 148 unchanged MLE rows and records 6,247 unresolved
rows. **No corrected full suite is evaluation-ready and no prose restoration
has yet been accepted on the actual evaluated rows.** Original inputs and scores
remain unchanged. No capacity, training, model inference or paid API was used.

| Source | Frozen rows | Ready | Unresolved |
|---|---:|---:|---:|
| AgentDojo | 1,500 | 0 | 1,500 |
| IaC | 1,443 | 0 | 1,443 |
| Rogue Eval | 1,499 | 0 | 1,499 |
| SHADE | 850 | 0 | 850 |
| CUA SHADE | 955 | 0 | 955 |
| MLE sabotage | 148 | 148 | 0 |

Receipts: `data/ood_visible_prose/restoration01/manifest.json`, `ready.jsonl`
and `unresolved.jsonl`. Ready records contain separately hashed trajectories and
teacher/student prompts; unresolved records retain parent identity and reason.
The ready file is a carry-forward subset, not a substitute OOD benchmark.
Canonical vs recorded source indices are kept separate. The first attempt caught
an index-join mistake before acceptance; subsequent preparation corrected the
join. Logs, including rejected preparation attempts, remain under
`results/ood_visible_prose/`.

Original-source research and concrete limitations are in the
[findings](../../docs/findings/ood_prose_restoration.md). The recovery code requires
checksum-bound typed source snapshots, exact roles/content/tool-call identity,
and well-formed hidden tags. CUA public exports must match all non-prose evidence.
It rejects semantic guessing, arbitrary plain-string overrides, malformed tags,
wrong rollouts and ambiguous source matches. The tests exercise restoration on
synthetic typed fixtures; they do not establish original-data recoverability.
Fourteen focused tests and Ruff checks pass. The local completion audit verifies
output checksums, unique IDs and exact 6,395-ID parent coverage. Upstream download
checksums are in `results/ood_visible_prose/upstream_checksums.json`.

## Original producer comparison

The subsequent independent export/comparison matches all 5,292 ControlArena OOD
rows against `Syghmon/blackboxmonitorsMATS@ccb77e37589baaee1f0e16ac9ab89ceac8f6cc8a`.
All ordered tool payloads and complete tool-result sequences match. Full role/
content streams match after removing leading empty think wrappers, assistant
outer whitespace and empty call-less assistant messages only. Nonempty reasoning,
inline literals and tool arguments are never removed by that normalization.

| Source | Rows | Exact full-content matches | Formatting-normalized matches |
|---|---:|---:|---:|
| AgentDojo | 1,500 | 982 | 1,500 |
| IaC | 1,443 | 807 | 1,443 |
| Rogue Eval | 1,499 | 999 | 1,499 |
| SHADE | 850 | 364 | 850 |

Normalized matching identifies a unique producer record for 5,290 rows. Two
AgentDojo rows have duplicate producer records with identical compared content;
no matched normalized content has conflicting labels.

The producer's 7,500 generic records still have untyped assistant strings. Their
Anthropic conversion contains text/tool-use/tool-result blocks, but no thinking
blocks, and uses generated tool IDs; these are converted exports rather than
evidence of original provider-private/public boundaries. No restoration was
accepted on that basis.

Two actual `.eval` logs from
`syghmon/scheming-exploration@00b143361b06852400a278dd2c477686daf4025a`
were independently exported: 257 attack and 257 honest samples from a separate
May 2025 Bash cohort. These contain 2,469 assistant messages, all string-valued;
they do not supply matched typed messages for our four OOD settings. The exports
retain native message objects, metadata, scores, archive member identity and
checksums. Three focused export/comparison tests pass; full receipts and matched
producer IDs are under `results/ood_visible_prose/original_control_comparison01/`.

The subsequent conversion audit verifies zero nonempty prose bodies across
161,413 assistant sections in the producer's own `formatted_transcript`.
All 5,292 matched frozen assistant/tool views equal that producer view, despite
separate system/user context edits. Omission therefore predates the Sinha
prepared benchmark. The producer's Anthropic-compatible text view matches its
generic assistant strings in 7,301 rows exactly; the other 199 differ only by
removal of whitespace-only blocks. None recovers original provider block types.
Four focused tests and lint pass for this extension. Source-search snapshots and
diagnostics remain under `results/ood_visible_prose/conversion_trace/`; the exact
native flattening/role-only formatter implementation was not found publicly.
