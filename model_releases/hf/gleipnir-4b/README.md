---
base_model: Qwen/Qwen3.5-4B
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
- base_model:adapter:Qwen/Qwen3.5-4B
model-index:
- name: Gleipnir 4B
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
      value: 0.919943
    - type: mean_ood_pauroc_at_20
      name: Mean-OOD pAUROC@20
      value: 0.810807
---

# Gleipnir 4B

Gleipnir 4B is a research monitor for AI control: it scores visible
trajectories of tool-using AI agents for sabotage, data exfiltration, and other
covert unauthorized actions taken while appearing to do an assigned task. It
is a LoRA adapter for `Qwen/Qwen3.5-4B`, not a standalone model.

Version 0.2 is trained on agent tool-use trajectories only, with the same data
and recipe as Gleipnir 27B. It is the exact weight average of three training
seeds (see Training).

This release is an early research artifact intended for reproducibility and
follow-up work. There is not yet a Gleipnir paper or blog post. It should not be
treated as a production safety system or as evidence that an interaction is
safe merely because the model assigns it a low score.

## Changelog

- **v0.2 (2026-10-20).** Retrained on the 7,836 strict-filtered tool
  trajectories only, dropping the 13,149 deception rows, so the 4B and 27B
  share one recipe. The released adapter is the exact average of three seeds
  that differ only in data order. Mean-OOD pAUROC@20 rises from 0.7824 to
  0.8108; inference cost and the scoring interface are unchanged. The adapter
  is now rank 384 (see Usage).
- **v0.1 (2026-10-18).** Single-seed adapter trained on 8,688 tool
  trajectories plus 13,149 deception examples. Pin revision `v0.1` to
  reproduce it.

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
interface as Gleipnir 9B and 27B.

## Results

Gleipnir 4B was evaluated once on the same frozen 6,395-row strict-OOD suite as
the other Gleipnir adapters, spanning AgentDojo, CUA-SHADE, IaC, MLE-Sabotage,
Rogue Eval, and SHADE-Arena. The primary metric is the unweighted mean across
sources of raw normalized partial AUROC through 20% false-positive rate.

| Model | Mean-OOD pAUROC@20 | Mean-OOD AUROC | Pooled pAUROC@20 | Pooled AUROC |
| --- | ---: | ---: | ---: | ---: |
| **Gleipnir 4B v0.2** | **0.8108** | **0.9199** | **0.7514** | **0.9335** |
| Gleipnir 4B v0.1 | 0.7824 | 0.9086 | 0.7015 | 0.9135 |
| Gleipnir 9B | 0.8212 | 0.9215 | 0.7837 | 0.9355 |
| Gleipnir 27B | 0.9087 | 0.9679 | 0.9234 | 0.9748 |

Per-source pAUROC@20: AgentDojo 0.9430, CUA-SHADE 0.7916, IaC 0.8845,
MLE-Sabotage 1.0000, Rogue Eval 0.9804, SHADE-Arena 0.2653.

- The release was chosen by a rule fixed before scoring, on held-out
  in-distribution data only: release the seed average if its macro
  pAUROC@20 on the 3,012-row ID set (Gloom-exfiltration and STRIDE test) is
  at least the three seeds' mean, otherwise the best single seed. The average
  reached 0.9124 against a seed mean of 0.9026 (0.9701 macro AUROC). The OOD
  suite was scored once and never used for selection.
- The individual seeds score 0.7869, 0.8125 and 0.7768 Mean-OOD pAUROC@20.
  The average is 1.9 points above their mean and the best of the four on
  AgentDojo and CUA-SHADE. Seed-to-seed OOD spread is large (standard
  deviation 1.8 points), which is why the release averages seeds rather than
  picking one.
- SHADE-Arena remains weak; long trajectories are the hardest case for this
  model size.
- The rows above compare released artifacts; they differ in base model, data
  mixture and seeds, so they are not a controlled model-size study.

## Training

Each seed was trained for one epoch on 7,836 action-only tool trajectories:
the 8,688 tool-trajectory training rows after removing those where Kimi K3's
decision disagrees with the source label. The sources are covert coding
sabotage (CoT-Red-Handed), shell sabotage and exfiltration (BashArena and
BashBench), and synthetic misaligned-agent scenarios (STRIDE and Gloom). No
deception-dataset rows were used. Training used only Kimi K3's soft probability
over the literal decision tokens; source labels were used for filtering,
provenance, and evaluation.

- Base: `Qwen/Qwen3.5-4B` at revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- Per seed: rank 128, alpha 256 LoRA on all attention and MLP projection
  modules; AdamW, learning rate `5e-5`, linear schedule with 3% warmup,
  effective batch 32, 245 updates; seeds 0, 1 and 2, differing only in data
  order, from the same initial adapter
- Context: up to 29,696 tokens, with zero audited truncation
- Training base: unquantized BF16 with FP32 master adapters, SDPA attention,
  16,384-token sequence packing; one NVIDIA B200, about an hour per seed
- Averaging: the three seed updates are combined exactly as one rank-384 LoRA
  (A matrices stacked, B matrices divided by 3, alpha 768), so the adapter
  equals the mean of the seed updates (maximum relative error 2.2e-8).
  Saved adapter: FP32, 509,607,936 parameters.

## Usage

The repository root contains the causal-LM PEFT master. The example below
reproduces the direct binary scoring interface; replace the example trajectory
with the complete visible interaction you want to monitor.

```python
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_id = "Jazhyc/Gleipnir-4B"
base_id = "Qwen/Qwen3.5-4B"
base_revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

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

Verify that each literal label tokenizes to exactly one distinct token.

The adapter is rank 384. For large evaluations the project merged the `vllm/`
key-rebased adapter into the base in FP32, exported BF16 weights, and served
them with vLLM 0.31. Master-to-serving parity passed (mean absolute score
difference 0.0022, correlation 0.9999), and the merged model has the same
inference cost as the base. Dynamic LoRA serving of a rank-384 adapter was not
tested; vLLM would need `--max-lora-rank 512`. Scores in this card were served
on an NVIDIA B300. Rescoring one seed there moved its ID pAUROC@20 by +0.0006
against the B200.

## Limitations and responsible use

- The average combines three seeds of one recipe. It was not optimized on
  multiple model families, languages, or deployment environments.
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
`experiments/seed_average/README.md`.

The adapter weights and release metadata are distributed under the MIT License;
see `LICENSE`. Users must also review and comply with the Qwen base-model terms,
the terms of the upstream datasets, and any applicable teacher/provider terms.
Several upstream source artifacts did not declare a license at collection time;
public weights do not grant rights to reconstruct or redistribute those
datasets.

## Citation

There is no paper or blog post to cite yet. For now, please cite the model page
and pin the Hugging Face repository revision used in your experiment.
