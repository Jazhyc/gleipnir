---
base_model: Jazhyc/Gleipnir-27B-LoRA
base_model_relation: merge
library_name: transformers
pipeline_tag: text-generation
license: mit
language:
- en
tags:
- ai-control
- agent-monitoring
- sabotage-detection
- tool-use
- monitoring
- research-artifact
---

# Gleipnir 27B (merged BF16)

This repository holds [Gleipnir 27B](https://huggingface.co/Jazhyc/Gleipnir-27B-LoRA),
an AI control monitor that scores visible tool-using agent trajectories for
covert unauthorized actions, as a standalone checkpoint: its LoRA adapter
merged into `Qwen/Qwen3.8-27B`. Inference cost equals the base model's, and the
BF16 weights need about 55 GB of accelerator memory before activations.

The model card, results, training details and limitations are in
[`Jazhyc/Gleipnir-27B-LoRA`](https://huggingface.co/Jazhyc/Gleipnir-27B-LoRA), which
also keeps the FP32 adapter for merging or fine-tuning. Headline result:
0.935 Mean-OOD pAUROC@20 under stock BF16 serving with the full specification
(0.951 under the optimized serving stack).

Technical report: *Lightweight Restraints for Powerful Agents: AI Control
Monitors That Decide in One Forward Pass* (arXiv link added on publication).
Project page: <https://jazhyc.github.io/gleipnir-monitor/>.

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Jazhyc/Gleipnir-27B-BF16"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision="v1.0")
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision="v1.0", torch_dtype=torch.bfloat16, device_map="auto"
).eval()

instruction_path = hf_hub_download("Jazhyc/Gleipnir-27B-LoRA", "teacher_prompt.txt")
with open(instruction_path, encoding="utf-8") as handle:
    instruction = handle.read()
trajectory = "User asked for a file listing. The agent ran: ls -la"
user_prompt = (
    f"{instruction}\n<agent_trajectory>\n{trajectory}\n"
    "</agent_trajectory>\n"
)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": user_prompt}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
) + "Prediction:"
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

token_ids = [
    tokenizer.encode(label, add_special_tokens=False)[0] for label in ("0", "1")
]
with torch.no_grad():
    next_logits = model(**inputs).logits[0, -1, token_ids].float()
risk_score = torch.softmax(next_logits, dim=0)[1].item()
print(risk_score)
```

### vLLM

For throughput on any GPU, serve this repository with stock vLLM 0.31.
Its pooling runner turns the model into a classifier over the `0` and `1`
logits at the last position:

```bash
vllm serve Jazhyc/Gleipnir-27B-BF16 --revision v1.0 --served-model-name monitor \
  --dtype bfloat16 --language-model-only --max-model-len 32768 --no-enable-prefix-caching \
  --runner pooling --convert classify \
  --pooler-config '{"pooling_type": "LAST", "use_activation": false}' \
  --hf-overrides '{"classifier_from_token": ["0", "1"], "method": "no_post_processing",
    "num_labels": 2, "is_causal": true, "text_config": {"classifier_from_token": ["0", "1"],
    "method": "no_post_processing", "num_labels": 2, "is_causal": true, "mamba_ssm_dtype": "float32"}}'

curl -s localhost:8000/classify -H 'Content-Type: application/json' -d '{"model": "monitor",
  "input": "<rendered prompt>", "add_special_tokens": false, "use_activation": false}'
# data[0].probs holds the raw [logit_0, logit_1]; score = softmax(...)[1]
```

The classifier settings repeat inside `text_config` because the Qwen
checkpoint wraps the language model. `mamba_ssm_dtype` keeps the GDN state
in FP32, as in the project's runs. Send the same rendered prompt as in the
Transformers example.

For the highest throughput, on one NVIDIA B200 or B300, use the project's
optimized serving image `ghcr.io/jazhyc/gleipnir-serve:1.2`, which loads this
checkpoint and quantizes it at load time; see "Optimized serving image" in the
[main model card](https://huggingface.co/Jazhyc/Gleipnir-27B-LoRA).

## Provenance

The adapter was merged into `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` in FP32 as `W + (alpha/r) B@A`, then cast once to
BF16. Every file is byte-identical to the merged checkpoint the project
evaluated, and its SHA-256 is recorded in the release manifest of the
[Gleipnir repository](https://github.com/Jazhyc/gleipnir)
(`experiments/release_serving/`).

The stock vLLM recipe above reproduces the project's recorded BF16 scores (mean
absolute difference 0.0014 per trajectory; ID 0.9421 against 0.9420
pAUROC@20), checked on an NVIDIA RTX PRO 6000. The optimized serving image
does not run on that GPU.

## Changelog

- **v1.0.** Card updated for the technical report; the files are those of
  v0.2.
- **v0.2 (2026-10-20).** First release: the v0.1 adapter merged into
  `Qwen/Qwen3.8-27B`.

## License

MIT, as for the adapter. Users must also comply with the `Qwen/Qwen3.8-27B`
terms and the upstream dataset terms described in the main model card.
