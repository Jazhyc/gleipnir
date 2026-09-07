# Reasoning in monitoring teacher and student training inputs

Status: completed 2026-09-07. Scope: all 8,688 monitoring training examples;
the 13,149 additional prior-deception examples in the standard mixed checkpoint
are outside this audit.

## Finding

The teacher receives the target agent's private reasoning and explicit thinking
tool calls. These also appear in the student's training input. This contradicts
the intended strict action-only input contract; it is not merely privileged
information supplied only to the teacher.

| Verified STRIDE reasoning form | Blocks/calls | STRIDE examples | Fraction of 1,350 |
| --- | ---: | ---: | ---: |
| Nonempty typed reasoning flattened into assistant text | 4,213 | 1,269 | 94.00% |
| Explicit `think` tool calls | 11,696 | 1,230 | 91.11% |
| Either form | — | 1,340 | 99.26% |

The union is 1,340/8,688 (15.42%) of the full monitoring training set. The
1,350 STRIDE training trajectories are unique by transcript hash.

No explicit `think` calls were found in Gloom (772 rows), BashArena (2,790),
BashBench (1,436) or CoT-Red-Handed (2,340). Those sources were not matched to
raw typed reasoning fields, so this is a confirmed lower bound for reasoning
exposure across the full mixture, not proof those sources are reasoning-free.

## Evidence and provenance

The actual teacher input file is
`data/tool_trajectory_monitoring/teacher_training/prompts.jsonl`, SHA-256
`c06833bc0c5e8893ea45e92ec642a71e751749ee1cd0796eb88f7ab38402763f`.
All 8,688 saved Kimi training-cache records across Makora, Morph and Fireworks
match its exact prompt hashes, with no missing or duplicate IDs. Thus the
reasoning-bearing prompts were actually submitted and scored.

All 8,688 monitoring examples in
`data/tool_trajectory_monitoring/distillation_scaling/mixed_student_rows.jsonl`
contain exactly the same trajectory strings under the compact student rubric.
This verifies actual student prompt text, rather than relying only on metadata.

For STRIDE, download the original `train1` and `train2` shards from
`aksh-n/rich-trajectories-cleaned@92a44c828935c325dc86cb390f874d0ee9345afb`.
They contain 947 and 946 original trajectories. All 1,350 selected training
examples match those sources by exact trajectory SHA-256. Inspect the matched
`trajectory_data.messages` for assistant blocks with `type=reasoning`, discard
empty blocks, and require each remaining block to occur verbatim in the actual
teacher input. All 4,213 do. Separately cross-check each row's serialized think
call count against structured `tool_calls[].function == "think"`; all counts
match. Source file SHA-256 values are:

- train1: `00487f025ecf195da96cd105685623373f78bdbfaf11f9a6cb8721c28650df2a`;
- train2: `3b355f67b48c0f25b20d05c6ca3c1faabbdcb86b6f12e0459aac6354f4545f86`.

The upstream source declares no license. Downloaded source shards remain in
ignored data storage for internal audit and are not redistributed.

## Mechanism and interpretation

`prepare_teacher_training_cache.extract_trajectory()` takes the user-message
payload from the paper author's three-message SFT rows. It excludes the outer
upstream monitor's assistant rationale and label completion. However, that user
payload already embeds the target agent's entire serialized transcript,
including its reasoning. `PromptTemplate.render()` preserves this text; it
performs no reasoning removal. The student materializer uses the same payload.

Kimi's `reasoning_effort=none` controls the teacher's own generation. It does
not remove reasoning already present in the input. Therefore the existing soft
targets are conditioned on these thoughts. Cleaning only student inputs would
create privileged-teacher supervision; a strictly action-only teacher condition
would require separately cleaned teacher prompts and new prompt-aware targets.
No training inputs or teacher caches were changed by this audit.

Reproduce with:

```bash
.venv/bin/python -m experiments.id_assistant_prose_audit.training_audit
```

The frozen artifact paths/checksums are in
`experiments/id_assistant_prose_audit/training_config.json`. Aggregate and
per-row findings are under `results/training_reasoning_audit/`. Focused tests
cover typed-reasoning extraction and distinguishing calls from tool-output
echoes; the full audit independently verifies source, cache and student joins.
