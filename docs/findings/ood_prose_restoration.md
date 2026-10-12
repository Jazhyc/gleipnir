# Recovering the original OOD public monitoring surface

The restoration request exposed an upstream recoverability gap: the available
ControlArena parquets flatten assistant content into strings, and the available
full-prose CUA exports do not match our frozen rollouts. The reproducible
[restoration preparation](../../experiments/ood_visible_prose/README.md) therefore
keeps 6,247 rows explicitly unresolved and carries forward only the 148 unchanged
MLE rows. It has not produced a corrected full suite or changed any reported
scores. This follows the [missing-prose audit](shade_missing_prose.md).

## What AISI's library tells us

The [ControlArena monitor reference](https://control-arena.aisi.org.uk/reference/monitor.html)
exposes tool-call, full-trajectory and CoT formatters. Full-trajectory formatters
preserve typed text and reasoning; monitoring information is configurable rather
than universally action-only. Blindly using the full formatter would change our
public-only information contract.

The source was pinned at
`UKGovernmentBEIS/control-arena@14909a39aefa56ff018619ed600e6cda32c65420`:

- `export/extractor.py` extracts `sample.messages` from original evaluation logs,
  preserving typed messages rather than reconstructing them from plain text.
- `settings/shade_arena/message_converter.py` passes Inspect's `.text` property
  to SHADE. Inspect's accessor returns strings or joins only `type: text` blocks;
  provider reasoning has a distinct representation. Its source was pinned at
  `UKGovernmentBEIS/inspect_ai@a1dbea8db3cc85e04ff532760d0eaaeb64df6d3a`.
- `monitor/_full_trajectory_prompt_formatter.py` handles `ContentText` and
  `ContentReasoning` separately. The generic full view includes reasoning, so it
  is not a public-text-only renderer.

This identifies how a correct restoration should use original typed content.
It does not prove which accessor the downstream author used to construct our
flattened strings, or recover the erased boundaries. Even text that sounds like
planning can be public; an admission of sabotage must not be deleted merely
because it is suspicious. Conversely, an untyped string cannot prove that an
apparent admission was publicly emitted rather than provider-private reasoning.

## Public upstream artifacts checked

The author precursor datasets were downloaded and inspected at these revisions:

| Dataset under `adityaasinha28/` | Revision | Rows |
|---|---|---:|
| `control_arena_agentdojo` | `a0f1e2131a8f27ebe14c285fa8ad34e439d128e2` | 1,500 |
| `control_arena_iac` | `cd0e7f667eba6214e2100f5fd30aec9f65aca1a8` | 1,500 |
| `control_arena_rogue_eval` | `7946e03ece9d8329a28c17d7b0fe778b9c774b39` | 1,500 |
| `control_arena_shade` | `7b5861e85f14381e70ed5b101fd7ceeec96d4f29` | 1,500 |
| `cua_shade_arena_aware` | `a968cce314ff353580628a0f920dda62acd5ba9e` | 292 |
| `cua_shade_arena_action_only_large` | `0d29d10d9dcefe3477d691d7a9f2e13b33e4939d` | 968 |

The four ControlArena precursors still contain untyped assistant strings.
The alternate `aksh-n/control_arena_rogue_eval` export at
`f30539753b13c2f99d11597ac0a0f065a81f19c9` also lacks typed boundaries.
The small action-only CUA precursor and the alternate large export were inspected
as well; these do not supply the missing original agent messages.

The 292-row full-prose CUA precursor does preserve assistant prose, but none of
our 955 CUA rows matches its full tool-output sequence. The stricter preparation
also requires matching user text, roles and tool names, and finds zero matches.
The small action-only precursor matches 270 of those full-prose traces, showing
that recovery is possible for that older cohort, not our frozen large cohort.

The creators' `scaleapi/mrt@3535620b2bdf0dc1e6140f6fbe198399aac2c74f`
points to `ScaleAI/mrt@bc636d0c3b881c7d8c76fd71b45b009c3710d437` for original
rollout JSONs (CC BY-NC 4.0). A same-task/model/index CUA example differs in tool
outputs and trajectory length. Sixteen benign variants of that example were
checked; none has its observed workspace output. Task name and nominal sample
index are insufficient identity. These public originals must not be substituted
for different evaluated rollouts, even when their task instructions resemble ours.

Source snapshots, metadata, checksums and match diagnostics are saved under
`data/ood_visible_prose/upstream/` and `results/ood_visible_prose/`.

## Remaining input requirement

### Original producer release and raw archives

A subsequent search found the underlying 7,500-trajectory producer release,
`Syghmon/blackboxmonitorsMATS@ccb77e37589baaee1f0e16ac9ab89ceac8f6cc8a`
(research-use-only). Its `control_arena.parquet` SHA256 is
`17897773c5e7bceae54195482b2389888a30e191b8a5182380d32e1994d165f9`.
An independent JSONL export and comparison links all 5,292 frozen ControlArena
OOD rows to that release. All ordered tool payloads/results match; the complete
role/content streams also match after narrowly specified formatting cleanup.
Exact vs normalized counts, source identifiers and qualifications are in the
[experiment record](../../experiments/ood_visible_prose/RESULTS.md#original-producer-comparison).

Thus the content retained in Sinha's `trajectory_data` matches the producer's
release; its evaluated `transcript` omits the assistant prose. This establishes
content lineage and the difference between the released views, not the exact
private/public boundary or the original exporter implementation. The producer's
generic messages are already strings. Its separately converted Anthropic view
does not recover native thinking types; API-compatible `type: text` annotations
must not be mistaken for original provider metadata.

The original producer's public
[scheming-exploration repository](https://github.com/syghmon/scheming-exploration/tree/00b143361b06852400a278dd2c477686daf4025a)
contains two real `.eval` archives. Independent native-sample export preserves
514 Bash samples and their messages without formatting/coercing content.
They are a different cohort, with string-valued assistant messages, rather than
the required matching OOD logs. The repository README identifies them as trusted
monitor Bash evaluations. No matching original OOD `.eval` archive was found in
the public sources inspected; this is a search result, not a claim that none
exists anywhere. Dataset environments and tasks are not a fixed rollout archive,
and rerunning them would generate new behavior rather than reconstruct old traces.

### Conversion-stage audit

The producer release itself has two different views: `messages` retains the
assistant strings; `formatted_transcript` removes their prose. Across all 7,500
producer rows, all 161,413 formatted assistant sections have empty prose bodies.
For each of the 5,292 matched OOD rows, the producer's formatted assistant/tool
sections are identical to Sinha's prepared assistant/tool sections. System and
user context are compared separately: downstream processing also changes that
context (including removal of continuation nudges and the Rogue Eval setting
envelope), so these are not claims of identical entire input strings.

This corrects an overly narrow attribution: the prose omission is already in
the original producer's action-only view and is not established as a new Sinha
preprocessing mistake. The producer's
[paper §4.1](https://arxiv.org/html/2603.00829) defines its primary monitoring
input in terms of queries, tool calls and outputs with agent reasoning removed.
The observed role-level omission is consistent with a broad implementation of
that action-only choice; it still removes useful public-facing reporting.

The Anthropic-compatible view rewraps the same generic assistant strings into
`type: text` blocks: 7,301 rows match exactly, and the other 199 differ only by
omission of whitespace-only text blocks. All 7,500 match after that exclusion.
These conversion annotations do not restore original native reasoning boundaries.

Public producer/coauthor repositories and preprocessing scripts were inspected
and pinned under `results/ood_visible_prose/conversion_trace/`. The public
`_inspect_export_to_parquet.py` found in the deliberative-monitor repository
exports judge scores rather than these rollout messages. No exact matching
native-message-to-string exporter or original action-only formatter source was
located in those public repositories. Thus the omission stage is established by
paired artifact evidence, while whether/how native block types were lost earlier
remains unresolved; they might already have been plain strings in the source.
The comparison entrypoint reproduces the field census and paired-view matches.

### Remaining source input

For the four ControlArena sources, obtain the original typed sample messages
from the producer's evaluation logs, or a documented export preserving the
same boundaries. For CUA, obtain full public transcripts for the large cohort
with exact matching non-prose evidence. The user does not have the original
`.eval` files. The library/docs establish the intended representation but cannot
regenerate a particular historical rollout or its original provider output.

The prepared recovery code joins by frozen source identity, verifies the local
original snapshot checksum, retains only typed public text, removes explicit
hidden blocks, and keeps frozen tool evidence. It preserves visible admissions,
reporting and message order. Unresolved rows are never silently dropped into a
smaller benchmark. Corrected full-suite readiness remains false until coverage
is complete. Recount token lengths and rescore changed inputs after recovery;
retain the existing frozen benchmark for historical comparisons.
