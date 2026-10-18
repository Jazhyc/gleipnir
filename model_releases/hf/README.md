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
