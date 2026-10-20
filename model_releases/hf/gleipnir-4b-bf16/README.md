---
base_model: Jazhyc/Gleipnir-4B
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

# Gleipnir 4B (merged BF16)

This repository holds [Gleipnir 4B](https://huggingface.co/Jazhyc/Gleipnir-4B),
a research monitor that scores visible tool-using agent trajectories for
covert unauthorized actions, as a standalone checkpoint. The v0.3 adapter (identical in v0.4)
was merged into `Qwen/Qwen3.5-4B` (revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) in FP32 as
`W + (alpha/r) B@A`, then cast once to BF16. Inference cost equals the base
model's, and the BF16 weights need about 9 GB of accelerator memory before
activations.

The model card, results, training details and limitations are in
[`Jazhyc/Gleipnir-4B`](https://huggingface.co/Jazhyc/Gleipnir-4B), which
also keeps the FP32 adapter for merging or fine-tuning. Headline result:
0.8039 Mean-OOD pAUROC@20 under stock BF16 serving (0.7703 under the optimized B200 stack).

The tag `v0.4` matches the adapter repository's release. Every file is
byte-identical to the merged checkpoint the project evaluated, and its SHA-256
is recorded in the release manifest of the
[Gleipnir repository](https://github.com/Jazhyc/gleipnir)
(`experiments/release_serving/`).

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Jazhyc/Gleipnir-4B-BF16"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision="v0.4")
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision="v0.4", torch_dtype=torch.bfloat16, device_map="auto"
).eval()

instruction_path = hf_hub_download("Jazhyc/Gleipnir-4B", "student_prompt.txt")
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

The project's stock BF16 results were served by vLLM 0.31 through its own
monitor-scoring endpoint, which reads the same two final-position logits.

For the highest throughput on an NVIDIA B200, use the project's optimized
serving image, which loads this checkpoint and quantizes it at load time; see
"Optimized B200 serving" in the
[main model card](https://huggingface.co/Jazhyc/Gleipnir-4B).

## License

MIT, as for the adapter. Users must also comply with the `Qwen/Qwen3.5-4B`
terms and the upstream dataset terms described in the main model card.
