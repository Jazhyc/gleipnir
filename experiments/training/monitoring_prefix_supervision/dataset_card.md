---
language:
- en
pretty_name: Gleipnir Prefix Teacher Cache
task_categories:
- text-classification
tags:
- monitoring
- distillation
- teacher-logprobs
configs:
- config_name: default
  data_files:
  - split: train
    path: logits.jsonl
---

# Gleipnir Prefix Teacher Cache

Research artifact containing **133,947 intermediate tool-trajectory predictions**
from `Qwen/Qwen3.5-27B-FP8`. This is a prediction cache, not a self-contained
trajectory dataset. It contains no trajectory text, original hard labels,
privileged rationales, or Kimi K3 full-trajectory targets.

## Important numerical limitation

**The completed cache failed its numerical-agreement audit.** On a fixed
64-prefix sample, fresh versus cached probability differences had mean absolute
error **0.010255635** and maximum **0.120145498**; five exceeded the predeclared
0.05 maximum limit. An earlier 24-prefix startup test had passed (mean 0.003717,
maximum 0.030967). Do not interpret that small passing test as validating this
full cache. The original failed audit and a five-case replay diagnostic are
included unchanged. Repeated fresh predictions matched exactly on those five
cases, while growing-prefix replay remained sensitive to execution history.

The project owner explicitly authorized an exploratory training screen using
these imperfect targets without recaching. That authorization is not evidence
that the numerical issue was fixed, or that the targets are calibrated. No
training benefit is established by this release. Not a production safety system.

## Contents and provenance

- `logits.jsonl`: one JSON object per intermediate boundary.
- `contract.json`: frozen model revision, runtime, prompt, source and reference hashes.
- `complete.json`: completed coverage and annotation runtime.
- `teacher_prefix.txt`: exact teacher instruction, evaluating behavior visible so far.
- `fresh_audit.json`, `failure_replay.json`: numerical reliability evidence.
- `SHA256SUMS.json`: checksums and sizes of release files.

The teacher revision is `97f5941bf617e31c5e237364a8602ce3f03a551a`.
Annotation used two H100 SXM5 GPUs, tensor parallelism 2, upstream FP8 weights,
BF16 model dtype, vLLM 0.28.0, Torch 2.13.0, Transformers 5.14.1, FlashInfer
GDN prefill, aligned prefix caching, and one active growing trajectory.
Runtime recurrent-state cache dtype was FP32. Responses were constrained to one
decision token (`0` or `1`), with both token logprobs explicitly requested.

Annotation completed on 2026-09-06 in 27,508 seconds: **1,199,840,658 input
tokens**, of which **1,065,849,568** were reported cached (88.83%). Source counts:
STRIDE 26,791; BashArena 72,894; BashBench 27,440; Gloom 6,822.
Final full-trajectory boundaries and statement-only turns were excluded; those
statements remain visible in later context. Twelve malformed boundaries were
not guessed. Format warnings are retained, including compressed step sequences.

## Schema and use

`logprob_0` and `logprob_1` are raw returned decision logprobs. `score` is the
binary-renormalized probability of problematic behavior:
`exp(logprob_1) / (exp(logprob_0) + exp(logprob_1))` (use a stable softmax).
These scores are teacher judgments, not ground truth.

`parent_prompt_id` and `lineage_group` identify the local parent/lineage;
`source` identifies the source family. These IDs are not universal upstream row
IDs. `prefix_ordinal` is zero-based; `prefix_count` counts eligible prefixes.
`end_character` is an exclusive Python-string character offset, not a byte or
token offset. `trajectory_sha256` and `prefix_sha256` identify exact UTF-8 text.
Other fields record rendered prompt hashes, contract identity, timestamps,
prompt/completion/cache token counts, and parser warnings.

To join against an independently obtained matching parent artifact:

1. Verify the parent-file SHA256 against `contract.json` before trusting local IDs.
2. Extract the text after `<agent_trajectory>\n` and before the final
   `</agent_trajectory>` in `student_prompt`, without stripping whitespace.
3. Verify `trajectory_sha256`, slice `trajectory[:end_character]`, then verify
   `prefix_sha256`. SHA256 inputs are UTF-8 encoded strings.
4. Render `teacher_prefix.txt.rstrip() + "\n<agent_trajectory>\n" + prefix`,
   add a newline if the prefix lacks one, then append `</agent_trajectory>\n`.
   Verify `rendered_user_prompt_sha256`.
5. Apply the pinned Qwen tokenizer's user chat template with
   `add_generation_prompt=True, enable_thinking=False`, append `Prediction:`,
   and verify `serving_prompt_sha256`.

The parent source is not distributed here; users without the matching source
cannot reconstruct trajectory text from these hashes. Preserve grouped splits
and avoid treating overlapping prefixes as independent evaluation examples.

## Rights and access

This release distributes derived numeric predictions and alignment metadata,
not the underlying source corpora. It does not grant rights to source datasets,
remove access restrictions, or relicense upstream material. No blanket license
for the underlying trajectories is asserted. Obtain source data separately and
follow its applicable terms. The original Kimi targets are not part of this release.
