---
base_model: Qwen/Qwen3.8-27B
library_name: peft
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
- knowledge-distillation
- lora
- research-artifact
- base_model:adapter:Qwen/Qwen3.8-27B
model-index:
- name: Gleipnir 27B
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
      value: 0.967861
    - type: mean_ood_pauroc_at_20
      name: Mean-OOD pAUROC@20
      value: 0.908739
---

# Gleipnir 27B

Gleipnir 27B is a research monitor for AI control: it scores visible
trajectories of tool-using AI agents for sabotage, data exfiltration, and other
covert unauthorized actions taken while appearing to do an assigned task. It
is a rank-128 LoRA adapter for `Qwen/Qwen3.8-27B`, not a standalone model.

Unlike Gleipnir 4B and 9B, it was not trained on general deception datasets;
its training data is agent tool-use trajectories only (see Training).

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
envelope is recorded in `prompt_contract.json`. It is the same instruction and
interface as Gleipnir 4B and 9B.

## Results

Gleipnir 27B was evaluated once on the same frozen 6,395-row strict-OOD suite as
the other Gleipnir adapters, spanning AgentDojo, CUA-SHADE, IaC, MLE-Sabotage,
Rogue Eval, and SHADE-Arena. The primary metric is the unweighted mean across
sources of raw normalized partial AUROC through 20% false-positive rate.

| Model | Mean-OOD pAUROC@20 | Mean-OOD AUROC | Pooled pAUROC@20 | Pooled AUROC |
| --- | ---: | ---: | ---: | ---: |
| **Gleipnir 27B** | **0.9087** | **0.9679** | **0.9234** | **0.9748** |
| Gleipnir 9B | 0.8212 | 0.9215 | 0.7837 | 0.9355 |
| Gleipnir 4B | 0.7824 | 0.9086 | 0.7015 | 0.9135 |
| Kimi K3 (teacher) | 0.9084 | 0.9616 | 0.9322 | 0.9776 |
| Qwen3.8-27B, unadapted | 0.8901 | 0.9535 | | |

Per-source pAUROC@20: AgentDojo 0.9983, CUA-SHADE 0.8806, IaC 0.9871,
MLE-Sabotage 1.0000, Rogue Eval 0.9989, SHADE-Arena 0.5875.

- Gleipnir 27B is level with its Kimi K3 teacher on Mean-OOD pAUROC@20
  (+0.0003) and above it on SHADE-Arena (0.5875 vs. 0.5745) and Rogue Eval.
  In the project's dated hosted-inference proxy it costs about 1/15 as much
  as Kimi K3 (`$1.69` vs. `$25.12` per 1,000 evaluations); that comparison is
  not a stable price quote.
- Training improved the unadapted base on every unsaturated source (+1.9
  points Mean-OOD pAUROC@20).
- On the project's held-out in-distribution set (3,012 Gloom-exfiltration and
  STRIDE test rows) it reaches 0.9217 macro pAUROC@20 and 0.9822 macro AUROC,
  against 0.8869 / 0.9730 for the unadapted base.

These are single-seed point estimates; the 4B student's OOD seed-to-seed
standard deviation is about 1.8 points. Gleipnir 27B was trained on a different
data mixture than the released 4B and 9B (see below), so the rows above compare
released artifacts, not a controlled model-size study.

## Training

Gleipnir 27B was trained for one epoch on 7,836 action-only tool trajectories:
the 8,688 tool-trajectory training rows after removing those where Kimi K3's
decision disagrees with the source label. The sources are covert coding
sabotage (CoT-Red-Handed, 1,949 rows), shell sabotage and exfiltration
(BashArena 2,504 and BashBench 1,348), and synthetic misaligned-agent scenarios
(STRIDE 1,310 and Gloom 725). It used no deception-dataset rows. Training used
only Kimi K3's soft probability over the literal decision tokens; source labels
were used for filtering, provenance, and evaluation.

- Base: `Qwen/Qwen3.8-27B` at revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Adapter: rank 128, alpha 256, all attention and MLP projection modules
  (gated-delta projections are not adapted)
- Optimization: AdamW, learning rate `5e-5`, linear schedule with 3% warmup,
  effective batch 32, seed 0, 245 updates
- Context: up to 29,696 tokens, with zero audited truncation
- Training base: unquantized BF16 with FP32 master adapters; segmented SDPA
  attention, 16,384-token sequence packing, gradient checkpointing of all
  gated-delta layers and 4 of 16 full-attention layers
- Saved adapter: FP32, 637,534,208 trainable parameters
- Hardware: one NVIDIA B200; training runtime about 4.2 hours

## Usage

The repository root contains the causal-LM PEFT master. The example below
reproduces the direct binary scoring interface; replace the example trajectory
with the complete visible interaction you want to monitor. The BF16 base needs
about 55 GB of accelerator memory before activations.

```python
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_id = "Jazhyc/Gleipnir-27B"
base_id = "Qwen/Qwen3.8-27B"
base_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

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
large frozen evaluations, the project merged the `vllm/` key-rebased adapter
into the base in FP32, exported BF16 weights, and served them with vLLM 0.31
after master-to-serving parity checks (correlation 0.9999).

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
`docs/findings/qwen38_27b_campaign.md` and
`experiments/qwen38_27b_campaign/README.md`.

The adapter weights and release metadata are distributed under the MIT License;
see `LICENSE`. Users must also review and comply with the Qwen base-model terms,
the terms of the upstream datasets, and any applicable teacher/provider terms.
Several upstream source artifacts did not declare a license at collection time;
public weights do not grant rights to reconstruct or redistribute those
datasets.

## Citation

There is no paper or blog post to cite yet. For now, please cite the model page
and pin the Hugging Face repository revision used in your experiment.
