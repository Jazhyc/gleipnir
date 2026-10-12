# Restore the original public monitoring surface

Status: deferred at the user's request. The
[comparability decision](../../docs/decisions/frozen_ood_comparability.md) retains
the frozen OOD benchmark and existing scores; this work remains a separate
diagnostic and must not replace evaluation defaults.

Purpose: recover omitted public assistant reports/commentary in the six frozen
OOD sources without supplying provider-private reasoning or attack instructions.
This is a data repair, not an objective/threshold experiment. Keep the original
6,395 IDs, labels, source membership and tool evidence. The frozen benchmark is
the comparator; corrected inputs get separate files and cache identities.

Acceptance: every restoration must bind to an original trace by message/tool
identity and a checksum-bound source snapshot. Typed reasoning is excluded;
explicit hidden-thinking blocks are removed from public text. Do not guess a
public/private split from wording or delete admissions merely because they look
suspicious. Flattened strings without recoverable boundaries are unresolved.
Stop before evaluation when any row remains unresolved. No GPU or paid inference
is needed to prepare the repair; this task does not authorize new capacity.

Run `.venv/bin/python -m experiments.ood_visible_prose.run`. Configuration is in
`config.json`; the output is `data/ood_visible_prose/restoration01/`, with ready
rows and unresolved receipts separated and an incomplete manifest when needed.
Unchanged MLE inputs can be carried forward. The remaining sources require
matched typed traces or verified original public transcript exports. The runner
checks the pinned full-prose CUA precursor in `cua_public_candidate` against the
frozen message/tool skeleton; same task names alone do not count as a match.

Optional `--originals PATH` reads JSONL records with `source`, `row_index`
(the zero-based filtered source position, not its upstream recorded index),
`messages` (original Inspect-style typed messages), and `provenance` containing
`source_uri`, `revision`, `license`, and `sha256`. `source_uri` must identify a
local file whose SHA256 matches; its JSON `messages` must equal the supplied
messages. The runner joins by frozen source/index and verifies raw identity,
roles, content collation and tool-call IDs/functions/arguments before using text
blocks. Strings in the flattened author export are never treated as typed proof.
Original snapshots and resulting inputs stay in ignored artifact directories.

Full-suite readiness, unresolved evidence, upstream searches and scientific
limitations are recorded in `RESULTS.md`. Do not compute a six-source average
from a smaller ready subset or reuse scores for changed prompts.

The original producer comparison runs with
`.venv/bin/python -m experiments.ood_visible_prose.compare_original_control`.
It exports the pinned producer parquet's messages to JSONL, compares each frozen
ControlArena row without label-based matching, and independently extracts native
sample JSON from the two public `.eval` archives under
`data/ood_visible_prose/original_logs/`. Outputs and source checksums are in
`results/ood_visible_prose/original_control_comparison01/`. This is an independent
native-message export, not ControlArena's provider-API formatter. The raw archives
are from a separate Bash cohort and cannot replace our four OOD sources.

The pinned CUA precursor can be downloaded without paid inference:

```bash
.venv/bin/hf download adityaasinha28/cua_shade_arena_aware \
  cua_shade_arena_aware.parquet --repo-type dataset \
  --revision a968cce314ff353580628a0f920dda62acd5ba9e \
  --local-dir data/ood_visible_prose/upstream
```
