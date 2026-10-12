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
