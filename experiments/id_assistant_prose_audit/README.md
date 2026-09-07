# ID assistant-prose removal audit

Hypothesis: removing assistant text outside tool calls reduces ID monitoring
input tokens. This is a descriptive token audit, not a model-quality experiment.

Intervention: remove prose-only assistant turns and prose preceding tool calls
from the frozen 3,012-row compact ID suite. A second variant also removes explicit `think` calls. Preserve all other tool
names and arguments (including submission text), results, user/system content,
row membership and labels. Parse the source-native line-delimited transcript headers. Retain the
assistant header for tool-bearing turns. Baseline: exact original prompts.
No OOD inputs or model predictions are used. No selection is performed.
Stop on source checksum, trajectory checksum, membership or format drift.

Count full inference inputs with the pinned Qwen3.5-4B tokenizer, frozen
non-thinking chat wrapper and `Prediction:` boundary, without truncation.
Report pooled and source full-input token totals for the original, prose-only
removal, and prose-plus-thinking removal variants. Derived
prompts have new hashes and explicit transformation provenance; old token/cache
metadata is retained only inside original_metadata. Original inputs stay frozen.

Run on CPU:

```bash
HF_HOME=/scratch/s4626451/.huggingface .venv/bin/python -m experiments.id_assistant_prose_audit.run
```

Outputs: `data/id_assistant_prose_audit/prompts.jsonl` and
`results/id_assistant_prose_audit/{summary.json,rows.jsonl}`. Transcript markers
are serialized text, not a lossless structured message format; quoted exact
role headers would be ambiguous. This audit uses the native header convention.


## Structured reasoning audit

Use the original STRIDE `trajectory_data` to locate nonempty typed `reasoning`
blocks and validate their verbatim presence in the frozen prompts. Verify none
remain in the cleaned prompts. Match every rendered STRIDE tool-call block
against structured source names/arguments before removing `think`: arguments
may contain literal newlines, nested code or malformed quote escaping, so a
line-based function-name heuristic is insufficient. The existing source renderer
uses JSON for dictionary arguments and trims trailing whitespace for strings.
All 19,735 rendered call blocks match this reconstruction exactly.

The STRIDE source revision and checksum are frozen in
`experiments/tool_trajectory_monitoring/prepare_qwen_reasoning_id_benchmark.py`:
SHA-256 `c13af47d00d3a32e9fcecb13df01e9f8bdf6a80e956d25ed00e8cb15dcc1d43b`.
Gloom's available source embeds a flat transcript, so typed private-reasoning
provenance is unavailable there. All its assistant prose is removed; this is
not a semantic guarantee that arbitrary tool arguments never contain reasoning.

No model run is launched. Prior frozen evaluation scores measure the original
inputs and cannot be treated as clean action-only results after this finding.

## Result

Completed 2026-09-07. Assistant-prose removal reduces full ID input tokens by
5.77%; additionally removing explicit think calls reduces them by 7.93% in total
(26.20% STRIDE, 4.63% Gloom). See
[the finding](../../docs/findings/id_assistant_prose_and_reasoning.md) for reasoning
prevalence, exact counts and interpretation.

## Training-input follow-up audit

Hypothesis: the same unintended reasoning exposure occurs in monitoring teacher
and student training inputs. This read-only audit covers all 8,688 monitoring
examples; the 13,149 additional prior-deception rows are outside its scope.
Do not modify training prompts or regenerate teacher labels. Stop on any source,
cache, trajectory or student-text mismatch. No model evaluation or selection is
part of this audit.

`training_config.json` freezes the teacher inputs, all three completed cache
shards, the mixed student file, and the original STRIDE train1/train2 shards.
The raw STRIDE files are downloaded from
`aksh-n/rich-trajectories-cleaned@92a44c828935c325dc86cb390f874d0ee9345afb`
into `data/training_reasoning_audit/source/raw_stride/data/`, preserving upstream
filenames. Match each selected STRIDE row by exact trajectory hash, then verify
raw typed reasoning verbatim in teacher inputs and cross-check serialized think
calls against structured source calls. Verify actual submitted cache hashes and
exact trajectory text in the mixed student inputs.

```bash
.venv/bin/python -m experiments.id_assistant_prose_audit.training_audit
```

Completed: 4,213 typed reasoning blocks in 1,269 STRIDE examples, plus 11,696
think calls in 1,230 STRIDE examples; the union is 1,340/1,350 STRIDE rows and
1,340/8,688 monitoring rows. All teacher-cache and student-input joins pass.
See [the finding](../../docs/findings/training_input_reasoning.md).
