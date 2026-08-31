---
base_model: Qwen/Qwen3.5-9B
library_name: peft
pipeline_tag: text-generation
license: mit
language:
- en
tags:
- ai-control
- deception-detection
- scheming-detection
- monitoring
- knowledge-distillation
- lora
- research-artifact
- base_model:adapter:Qwen/Qwen3.5-9B
model-index:
- name: Gleipnir 9B
  results:
  - task:
      type: text-classification
      name: Binary AI-control monitoring
    dataset:
      type: gleipnir-six-source-strict-ood
      name: Six-source strict-OOD action-only tool-trajectory suite
    metrics:
    - type: roc_auc
      name: Mean-OOD AUROC
      value: 0.921536
    - type: mean_ood_pauroc_at_20
      name: Mean-OOD pAUROC@20
      value: 0.821183
---

# Gleipnir 9B

Gleipnir 9B is a research monitor for detecting deception, scheming, and other
control-relevant problematic behavior in visible AI-agent trajectories. It is
a rank-128 LoRA adapter for `Qwen/Qwen3.5-9B`, not a standalone model.

This release is an early research artifact intended for reproducibility and
follow-up work. There is not yet a Gleipnir paper or blog post. It should not be
treated as a production safety system or as evidence that an interaction is
safe merely because the model assigns it a low score.

## What the model scores

The model receives an action-only transcript: visible messages, tool calls,
commands, outputs, edits, and final reports, without access to private chain of
thought. At the frozen decision boundary it compares the next-token logits for
literal `0` and `1`:

- `Prediction:1`: a concrete unauthorized or harmful action, attempted
  safeguard bypass, compliance with a conflicting instruction in untrusted
  input, or material deceptive concealment;
- `Prediction:0`: task-serving behavior, including honest mistakes, failed
  attempts, harmless exploration, incomplete work, and reasonable adaptation.

Use the normalized `1` versus `0` logit as a continuous risk score. The exact
compact instruction is included in `student_prompt.txt`, and its complete
envelope is recorded in `prompt_contract.json`.

## Results

Both released Gleipnir adapters were evaluated once on the same frozen 6,395-row
strict-OOD suite spanning AgentDojo, CUA-SHADE, IaC, MLE-Sabotage, Rogue Eval,
and SHADE-Arena. The primary metric is the unweighted mean across sources of raw
normalized partial AUROC through 20% false-positive rate.

| Model | Mean-OOD pAUROC@20 | Mean-OOD AUROC | Pooled pAUROC@20 | Pooled AUROC |
| --- | ---: | ---: | ---: | ---: |
| Gleipnir 4B | 0.7824 | 0.9086 | 0.7015 | 0.9135 |
| **Gleipnir 9B** | **0.8212** | **0.9215** | **0.7837** | **0.9355** |

Gleipnir 9B leads Gleipnir 4B by 0.0388 Mean-OOD pAUROC@20 and 0.0130
Mean-OOD AUROC. In the project's dated hosted-inference proxy it costs about
three times as much, so the 4B release remains a strong lower-cost point. These
are single-seed point estimates, and the hosted-cost comparison is not a stable
price quote.

Adding 13,149 prior deception examples to the 8,688 matched tool-trajectory
examples was macro-neutral relative to the matched-only 9B adapter: Mean-OOD
pAUROC@20 changed by -0.0015 and Mean-OOD AUROC by +0.00003. Pooled AUROC rose
by 0.0181, but that does not establish a general per-source transfer benefit.

## Training

Gleipnir 9B was trained for one epoch on 21,837 examples: 8,688 action-only tool
trajectories and 13,149 prior deception examples from the Aletheia's Quest and
Liars' Bench line. Training used only Kimi K3's soft probability over the
literal decision tokens. Source hard labels were retained for provenance and
evaluation but had zero loss weight.

- Base: `Qwen/Qwen3.5-9B` at revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Adapter: rank 128, alpha 256, all attention and MLP projection modules
- Optimization: AdamW, learning rate `5e-5`, effective batch 32, seed 0
- Context: up to 29,696 tokens, with zero audited truncation
- Training base: 4-bit NF4 with double quantization and BF16 compute
- Saved adapter: FP32, 232,783,872 trainable parameters
- Hardware: one NVIDIA H100 80 GB; training runtime about 6.65 hours

## Usage

The repository root contains the causal-LM PEFT master. The example below
reproduces the direct binary scoring interface; replace the example trajectory
with the complete visible interaction you want to monitor.

```python
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_id = "Jazhyc/Gleipnir-9B"
base_id = "Qwen/Qwen3.5-9B"
base_revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"

tokenizer = AutoTokenizer.from_pretrained(base_id, revision=base_revision)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    revision=base_revision,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_id).eval()

instruction_path = hf_hub_download(adapter_id, "student_prompt.txt")
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
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

token_ids = [
    tokenizer.encode(label, add_special_tokens=False)[0] for label in ("0", "1")
]
with torch.no_grad():
    next_logits = model(**inputs).logits[0, -1, token_ids].float()
risk_score = torch.softmax(next_logits, dim=0)[1].item()
print(risk_score)
```

Verify that each literal label tokenizes to exactly one distinct token. For
large frozen evaluations, the `vllm/` subdirectory contains the key-rebased
adapter layout used by the project's parity-gated vLLM 0.24.0 campaign.

## Limitations and responsible use

- The checkpoint has one training seed and was not optimized on multiple model
  families, languages, or deployment environments.
- It can miss subtle harmful behavior and can flag benign but unusual actions.
  Its score is not calibrated for a new deployment without local validation.
- It observes actions, not hidden intent or private reasoning. Absence of
  visible evidence is not evidence of aligned intent.
- Long trajectories, prompt injections inside transcripts, source-specific
  artifacts, and changes to chat templates or scoring boundaries may shift its
  behavior.
- The benchmark measures ranking. The `0.5` threshold is an interface default,
  not a recommended production operating point.
- Do not use it as the sole basis for punitive decisions about people, or as the
  only safeguard around capable autonomous systems.

Use layered controls, preserve the continuous score, validate by task/source
and model family, and keep a human-review or defer path for ambiguous cases.

## Artifacts, provenance, and terms

This repository does not redistribute training prompts, trajectories, source
labels, teacher caches, or benchmark rows. `release_manifest.json` records the
adapter, prompt, base-revision, and packaging checksums needed to audit this
release. The full experiment design and result audit are in the
[Gleipnir repository](https://github.com/Jazhyc/gleipnir), especially
`docs/research/tool_trajectory_distillation_ood_scaling.md`.

The adapter weights and release metadata are distributed under the MIT License;
see `LICENSE`. Users must also review and comply with the Qwen base-model terms,
the terms of the upstream datasets, and any applicable teacher/provider terms.
Several upstream source artifacts did not declare a license at collection time;
public weights do not grant rights to reconstruct or redistribute those
datasets.

## Citation

There is no paper or blog post to cite yet. For now, please cite the model page
and pin the Hugging Face repository revision used in your experiment.
