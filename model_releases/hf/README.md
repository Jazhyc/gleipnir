# Hugging Face adapter releases

This directory holds the reviewed, tracked inputs for the Gleipnir 4B, 9B
and 27B Hugging Face releases. The large model files remain under ignored
`results/` paths and are copied into an ignored staging directory only after
their checksums and adapter contracts pass.

Prepare the release directories with (`--model 4b|9b|27b` for one):

```bash
python scripts/prepare_hf_adapter_release.py
```

This creates `results/huggingface_release/Gleipnir-{4B,9B,27B}/`. A model whose
recipe differs from the shared `training` block (the 27B: strict-filtered
tool trajectories only, BF16 base) overrides those fields in its own
`training` entry. Each directory contains:

- the FP32 causal-LM PEFT adapter at the repository root;
- the parity-checked Qwen image-text/vLLM key layout under `vllm/`;
- the exact compact student prompt and prompt contract;
- a concise training summary and a checksum-complete release manifest; and
- the corresponding reviewed model card.

Inspect those staged trees before publishing. Uploading is intentionally a
separate, explicit operation because it creates public external repositories:

```bash
python scripts/upload_hf_adapter_release.py \
  --release-dir results/huggingface_release \
  --namespace Jazhyc
```

Use `--private` for a private canary. The uploader verifies every staged file
against `release_manifest.json` before contacting Hugging Face and refuses to
reuse a pre-existing repository unless `--allow-existing` is given.

The adapter weights and release metadata are distributed under the MIT License,
which is included in each staged repository. The Qwen base-model terms and the
terms of the upstream training data and teacher provider still apply.

Published: `Jazhyc/Gleipnir-27B` (2026-10-18; public), in the Efficient Control
Monitoring collection after Gleipnir 4B and 9B. First upload `7a932209`; the
card was then corrected to describe what it is trained for (tool-use agent
sabotage/exfiltration monitoring, no deception datasets) at `e5c4a7a6`, with
all 12 files verified against `release_manifest.json` on the Hub.

Versioning: the repos are research previews. Each was tagged `v0.1` on
2026-10-18 at its current commit (4B `411bdb7c`, 9B `bd86dda2`, 27B
`e5c4a7a6`). Retrained adapters on the same base go to the same repo with a
card changelog and the next `v0.x` tag; a public launch rewrites the cards and
tags `v1.0`; a different base model gets a new repo name.

Gleipnir 4B v0.2 (2026-10-20): the exact three-seed weight average of the
strict-filtered tool-trajectory-only recipe shared with the 27B
([seed average](../../experiments/seed_average/README.md)). It is a rank-384
adapter, so its config sets `lora_rank: 384`. It was staged with
`--output-dir results/huggingface_release_v0.2` and uploaded with
`--allow-existing` as commit `7776a3e3`. All 10 manifest files were verified
on the Hub, and the commit is tagged `v0.2`; `v0.1` still pins the mixed-data
adapter. The 9B and 27B cards still list the v0.1 4B numbers in their
comparison tables.

Card update (2026-10-20, user request), with no new tags because the weights
are unchanged:
- **Gleipnir 27B** recommends and reports the full teacher prompt (0.9352
  Mean-OOD, 0.9420 CoT-removed ID pAUROC@20; [decision](../../docs/decisions.md)).
  Its config sets `evaluation_prompt: teacher`, so `teacher_prompt.txt` and the
  teacher template hash are staged with it. Commit `694b5239`; 11 files
  verified.
- **Gleipnir 4B**'s comparison row for the 27B now shows those numbers.
  Commit `c20114a1`; 10 files verified. `v0.2` still points to the release
  commit.
- **Gleipnir 9B** is marked superseded by Gleipnir 4B v0.2 (user request):
  similar OOD quality (0.8212 against 0.8108 pAUROC@20, within seed noise) at
  about 3.3 times the hosted cost. It is kept as a research artifact, and its
  comparison table is refreshed. Commit `9c4c246e`; 10 files verified.

Gleipnir 4B v0.3 (2026-10-20, user request): the v0.2 average plus same-model
QAD for optimized serving ([QAD](../../experiments/qad_v02/README.md)).
- **Source.** The release adapter is the QAD master (`0716efc1…`) in the
  campaign-layout copy `results/qad-v02-01/4b/monitor`. Its
  `training_metadata.json` was added there to record the base, the QAD
  receipt and the starting adapter.
- **Results.** The card reports BF16 serving (0.8039 OOD, 0.9332 ID
  pAUROC@20) and optimized serving (0.7703 OOD, 0.9117 ID, 1.89 times BF16
  throughput).
- **Publication.** Staged at `results/huggingface_release_v0.3` and uploaded
  as commit `e36aa2bb`. All 10 files were verified on the Hub, with no stale
  files, and the commit is tagged `v0.3`; `v0.1` and `v0.2` remain pinnable.
