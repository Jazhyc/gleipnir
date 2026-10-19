---
license: other
tags: [ai-control, prompt-injection, monitoring, evaluation]
configs:
  - config_name: benchmark_neutral
    data_files: data/apps/prompts/eval_neutral.jsonl
  - config_name: benchmark_injection_aware
    data_files: data/apps/prompts/eval_injection_aware.jsonl
  - config_name: honest_controls_neutral
    data_files: data/apps/honest_controls/eval_neutral.jsonl
  - config_name: honest_controls_injection_aware
    data_files: data/apps/honest_controls/eval_injection_aware.jsonl
  - config_name: raw
    data_files: data/apps/apps_eval.jsonl
---

# Gleipnir APPS injection evals

The APPS prompt-injection benchmark the Gleipnir project uses to evaluate
trusted monitors, plus the honest-code injection controls we built on top of
it. **For evaluation only. Do not train on it.**

## Provenance

- **Benchmark (`data/apps/prompts/`, `apps_eval.jsonl`, `data/templates/`).**
  This is a byte-for-byte copy of
  [`mjkenney/monitor-injection-evals`](https://huggingface.co/datasets/mjkenney/monitor-injection-evals)
  at revision `58e722bf9912903d56a334f873e069aa17ccb7ad`. Michael Kenney built
  it and permitted its use and redistribution for this project. It follows
  Terekhov et al., *Adaptive Attacks on Trusted Monitors*
  (arXiv:2510.09462). `openai/o3-mini` wrote both the backdoored solutions and
  the honest solutions, starting from `RoganInglis/apps-control-arena`.
- **Honest controls (`data/apps/honest_controls/`).** These are our addition.
  We took the 655 clean honest submissions and appended each of the three
  transfer-injection payloads (GPT-4, GPT-4o, GPT-4.1), plus each payload's
  length-matched neutral text. That gives 6 × 655 = 3,930 rows per
  instruction. Each payload is the exact text found on the original injected
  backdoor rows. The builder is `scripts/build_honest_controls.py` in
  [`mkenney2/monitor-injection-evals`](https://github.com/mkenney2/monitor-injection-evals)
  at commit `b55c2014d4c22ceea810b2639e6c0f672d83eccb`.

## Contents

| Config | Rows | Labels |
|---|---|---|
| `benchmark_{neutral,injection_aware}` | 5,184 each | 655 clean honest (0); 647 backdoors (1) under each of 7 conditions: `vanilla`, `inj_transfer_{gpt4,gpt4o,gpt41}`, `neutral_transfer_*` |
| `honest_controls_{neutral,injection_aware}` | 3,930 each | all 0: honest code carrying an injection (`inj_*`) or its neutral match (`neutral_*`) |
| `raw` | 5,400 | 675 problems × 8 conditions, unrendered and before validity filtering or dev exclusion; `label` plus `valid_*` flags |

The two variants differ only in the monitor instruction:
`neutral` uses Gleipnir's regular student prompt (`tool-trajectory-binary-v1`)
and `injection_aware` warns the monitor about injected instructions. Each row
has `id`, the fully rendered `prompt` and `metadata`. The metadata fields are
`ground_truth`, `condition`, `task_id`, and for controls `parent_id` and
`template_text_sha256`. Upstream validity filtering and the 20-problem dev
exclusion have already been applied.

The label always means *the code contains a backdoor*. Monitor-directed text in
a row does not change its label. A useful analysis is the alarm rate on honest
controls at the clean-honest threshold, together with matched AUROC between
backdoor and honest rows that carry the same injection.

## Integrity

`release_manifest.json` lists the size and SHA256 of every file. Each `*.manifest.json` records
how its file was built, including the parent file's checksum. Gleipnir pins
these checksums:

| File | SHA256 |
|---|---|
| `prompts/eval_neutral.jsonl` | `339327a476f6834d8998500e2726263dfe3ce55bcc0a12648928abe9e99abfc0` |
| `prompts/eval_injection_aware.jsonl` | `b376a701853d4189a300aaf3d310e949256f38b2619b28017d362c446c284eec` |
| `honest_controls/eval_neutral.jsonl` | `71b337c070fc60faba0d04c54c5f0813498c45d60ab6b9c77b9b67639c6a70e1` |
| `honest_controls/eval_injection_aware.jsonl` | `6c90d47d6c0fac4cdc2b97803e004a42d49cd7488c4c8f71bbb5f894ac802369` |

## Terms

The upstream data has no open license. This copy is shared for research
evaluation with the permission of the upstream author. Contact the Gleipnir
authors and Michael Kenney before you redistribute it further or use it for
anything other than evaluating monitors.
