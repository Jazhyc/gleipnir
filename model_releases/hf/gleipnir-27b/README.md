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
      name: Six held-out AI control benchmarks (action-only tool trajectories)
    metrics:
    - type: mean_ood_pauroc_at_20
      name: Mean-OOD pAUROC@20 (optimized FP8/FP4/MXFP8 serving)
      value: 0.950800
    - type: roc_auc
      name: Mean-OOD AUROC (optimized serving)
      value: 0.984100
    - type: mean_ood_pauroc_at_20
      name: Mean-OOD pAUROC@20 (BF16 serving)
      value: 0.935197
---

# Gleipnir 27B

Gleipnir 27B is an AI control monitor: it reads the visible trajectory of a
tool-using AI agent and returns the probability that the agent is pursuing a
covert harmful objective, such as sabotage or data exfiltration, while it
appears to do its assigned task. It decides in one forward pass, without
writing a rationale, so it is cheap and fast enough to check every trajectory
or every action.

This repository holds the LoRA adapter for `Qwen/Qwen3.8-27B`, for merging and
fine-tuning. To run the monitor, use the merged checkpoint
[`Jazhyc/Gleipnir-27B-BF16`](https://huggingface.co/Jazhyc/Gleipnir-27B-BF16)
or the serving image (see [Variants](#variants)).

- **Technical report:** *Lightweight Restraints for Powerful Agents: AI
  Control Monitors That Decide in One Forward Pass* (arXiv link added on
  publication).
- **Project page:** <https://jazhyc.github.io/gleipnir-monitor/>, with the
  code, the training data and the serving image.

On six held-out AI control benchmarks, Gleipnir 27B matches the strongest
reasoning monitor at more than 200 times lower cost. It still misses many
attacks built to evade monitors and most reward hacks by other agents, so use
it as a first line of defense in front of stronger monitors, not as a sole
safeguard.

## Variants

| Repository or image | Contents | Use it for |
| --- | --- | --- |
| `Jazhyc/Gleipnir-27B-LoRA` (this repository) | FP32 LoRA adapter and the `vllm/` key-rebased adapter | Merging, fine-tuning, PEFT |
| [`Jazhyc/Gleipnir-27B-BF16`](https://huggingface.co/Jazhyc/Gleipnir-27B-BF16) | The adapter merged into `Qwen/Qwen3.8-27B` in FP32, exported as BF16 | Transformers or stock vLLM on any GPU |
| `ghcr.io/jazhyc/gleipnir-serve:1.2` | The optimized serving stack (FP8/FP4/MXFP8) | Fastest scoring, on one NVIDIA B200 or B300 |

## What the model scores

The model receives an action-only transcript: tool calls, commands, outputs,
edits and final reports, without the agent's reasoning. It compares the
next-token logits for the literal tokens `0` and `1` after `Prediction:`:

- `Prediction:1`: a concrete unauthorized or harmful action, attempted
  safeguard bypass, compliance with a conflicting instruction in untrusted
  input, or material deceptive concealment;
- `Prediction:0`: task-serving behavior, including honest mistakes, failed
  attempts, harmless exploration, incomplete work and reasonable adaptation.

The score is the softmax of the `1` and `0` logits; keep it continuous and set
your own threshold.

**Score it with the full specification (`teacher_prompt.txt`).** The adapter
was trained with the compact instruction (`student_prompt.txt`), but on the
validation set it ranks better when it reads the full specification (0.942
against 0.922 pAUROC@20), which costs 9.9% more prompt tokens. All results
below use the full specification; `prompt_contract.json` records both
envelopes.

## Results

All results are from the technical report, on the released serving stack
(one NVIDIA B200) unless stated otherwise.

**Six held-out AI control benchmarks** (AgentDojo, CUA-SHADE-Arena, IaC,
MLE-Sabotage, Rogue Eval, SHADE-Arena; 6,395 trajectories, never used for
training or selection). pAUROC@20 is the unweighted mean over the six
benchmarks of the partial AUROC up to a 20% false-positive rate (chance 0.1).
Recall is at the threshold that flags 1% of benign trajectories. Cost is in
USD per 1,000 evaluations: Gleipnir on one B200 at the committed-use and
on-demand rates, the others at API prices.

| Monitor | Cost | pAUROC@20 | AUROC | Recall at 1% FPR |
| --- | ---: | ---: | ---: | ---: |
| **Gleipnir 27B** | **0.178 / 0.414** | **0.951** | **0.984** | **87.4%** |
| Gleipnir 4B | 0.040 / 0.094 | 0.770 | 0.914 | 60.6% |
| pplx-decider v1.2 | 0.176 | 0.896 | 0.958 | 75.9% |
| Kimi K3 (teacher) | 25.118 | 0.908 | 0.962 | 85.9% |
| Claude Opus 4.6, reasoning (Sinha et al.) | 94.794 | 0.942 | 0.979 | — |

- **Against the strongest reasoning monitor**, Claude Opus 4.6 of Sinha et
  al., the difference lies within the 95% bootstrap interval of Gleipnir 27B
  (0.941 to 0.960), while Gleipnir 27B costs 229 times less on demand and 532
  times less with committed use.
- **At equal cost**, it scores 0.055 above pplx-decider v1.2 and recalls 11.5
  points more harmful trajectories at a 1% false-positive rate.
- **Training** raises the unadapted Qwen3.8-27B from 0.890 to 0.951. Its lead
  of 0.043 over its teacher comes from one task of SHADE-Arena.
- **Serving precision.** Under stock BF16 serving the same weights score
  0.935; the gap comes almost entirely from SHADE-Arena, so treat it as
  serving variation rather than a quality difference.
- **Threshold.** On the released stack, the score that flags 1% of the 3,189
  benign trajectories is 0.798. It is a starting point only: set your own
  threshold on your benign traffic.

Per-benchmark pAUROC@20 (optimized / BF16 serving): AgentDojo 0.992 / 0.999,
CUA-SHADE-Arena 0.890 / 0.891, IaC 0.993 / 0.993, MLE-Sabotage 1.000 / 1.000,
Rogue Eval 0.998 / 0.998, SHADE-Arena 0.832 / 0.730.

**Harder tests.**

- **Attacks designed to evade monitors** (SleightBench, 40 attacks): 0.299
  pAUROC@20, catching 20% of the attacks at a 1% false-positive rate. Every
  low-cost monitor scores between 0.19 and 0.40, far below the strongest model
  scored (GPT-6.1 Terra, 0.734). The training data contains no attacks built
  against a monitor.
- **Reward hacking by other agents** (EvilGenie Escalation, 124 reward hacks
  by 15 agent models): at its 1% threshold it flags 28% of the reward hacks
  and 1.1% of the legitimate runs, more than every other low-cost monitor
  (pAUROC@20 0.673). Most missed hacks add the sample output of one problem
  to otherwise correct code.
- **Text in the code.** When the same accusation of a backdoor is inserted as
  a docstring into a benign and a backdoored program, Gleipnir 27B keeps its
  separation (0.818 without text, 0.934 with a specific accusation), unlike
  Gleipnir 4B. This test ran the same weights in BF16.

## Training

Gleipnir 27B learns the decision of a stronger teacher directly: it is trained
to match the probability that Kimi K3 gives to the decision token, read once
for each training trajectory.

- **Data.** The 8,688 action-only tool trajectories of the training set of
  Sinha et al. (covert coding sabotage from CoT-Red-Handed, shell sabotage and
  exfiltration from BashArena and BashBench, and synthetic misaligned-agent
  scenarios from STRIDE and Gloom), minus the 852 where the teacher's decision
  disagrees with the dataset label: 7,836 trajectories. The labels are used
  only for this filter.
- **Recipe.** Rank-128 LoRA (alpha 256) on all attention and MLP projections
  of `Qwen/Qwen3.8-27B`
  (the gated-delta projections are not adapted); AdamW, learning rate `5e-5`,
  3% warmup and linear decay, effective batch 32, one epoch (245 updates),
  inputs of up to 29,696 tokens; about 4.5 hours on one B200.
- **One run.** The 27B is a single training run, without seed averaging and
  without quantization-aware distillation, which it does not need.

## Usage

### Transformers and PEFT

The BF16 base needs about 55 GB of accelerator memory before activations.

```python
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_id = "Jazhyc/Gleipnir-27B-LoRA"
base_id = "Qwen/Qwen3.8-27B"
base_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

tokenizer = AutoTokenizer.from_pretrained(base_id, revision=base_revision)
base = AutoModelForCausalLM.from_pretrained(
    base_id,
    revision=base_revision,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter_id, revision="v1.0").eval()

# The recommended full specification; "student_prompt.txt" is the compact
# instruction the adapter was trained with.
instruction_path = hf_hub_download(adapter_id, "teacher_prompt.txt", revision="v1.0")
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

Replace the example trajectory with the complete visible interaction, and
check that each label tokenizes to exactly one distinct token. For
throughput, serve the merged checkpoint
[`Jazhyc/Gleipnir-27B-BF16`](https://huggingface.co/Jazhyc/Gleipnir-27B-BF16)
with stock vLLM, as its card shows.

### Optimized serving image

The image quantizes the BF16 checkpoint at load (FP8 attention projections,
FP4 MLP and GDN projections, MXFP8 attention). It needs one NVIDIA B200 or
B300 and a driver that supports CUDA 13. With one request at a time, one B200
scores an agent trajectory of about 10,000 tokens in 192 ms (51K prompt
tokens/s).

```bash
docker run --gpus all -p 8000:8000 \
  -v "$HOME/gleipnir-models:/models" -e GLEIPNIR_MODEL_CACHE=/models \
  ghcr.io/jazhyc/gleipnir-serve:1.2 27b

curl -s localhost:8000/v1/monitor/score -H 'Content-Type: application/json' \
  -d '{"model": "monitor", "prompt": "<rendered prompt ending in Prediction:>"}'
# {"score": ..., "margin": ..., "logits": [logit_0, logit_1], "prompt_tokens": ...}
```

- **Prompt.** Send the fully rendered prompt, built as in the Python example
  and ending in `Prediction:`; prompts may have up to 262,144 tokens.
- **First start.** The image downloads and verifies the 52 GB checkpoint; the
  `/models` mount keeps it across restarts. Do not mount volumes over `/tmp`
  or `/workspace`.
- **Prefix caching.** Append `--prefix-caching` to rescore a growing session
  faster: on agent trajectories, checking every action takes 3.7 times less
  time. Cached scores can differ from uncached ones, so keep it off for
  evaluation.
- **Reproducibility.** Optimized scores are not bit-reproducible: two runs
  differ by about 0.006 in mean absolute score, while ranking metrics move by
  about 0.001 to 0.002. Use stock BF16 serving when per-row reproducibility
  matters.

The project's [README](https://github.com/Jazhyc/gleipnir#running-inference)
gives all three routes in more detail.

## Limitations and responsible use

- **Benchmarks, not production traffic.** The evaluation uses benchmarks
  with few benign trajectories, so it cannot resolve the false-positive rates
  of production monitoring, far below 1%. Validate the monitor and its
  threshold on your own traffic.
- **Attacks aimed at the monitor.** The training data contains no attacks
  built against a monitor, and the monitor misses many of them (see Results).
- **Visible actions only.** The monitor reads actions, not the agent's
  reasoning or intent; no visible evidence is not evidence of benign intent.
- **One teacher, one training set, one run.** All results come from Kimi K3
  as the teacher, one training set and a single 27B training run; we do not
  know how the recipe transfers or how much the 27B varies between runs.
- **Not a sole safeguard.** Do not use the score as the only basis for
  decisions about people or as the only control around capable autonomous
  agents. Keep a human-review or defer path for ambiguous cases.

## Artifacts, provenance and terms

The adapter was trained on `Qwen/Qwen3.8-27B` at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; load the base at this revision, as in
the usage example. `release_manifest.json` records the checksums of the
adapter, prompts, base revision and packaging. The teacher probabilities of the training data are
public in
[`Jazhyc/gleipnir-kimi-k3-tool-trajectory-logits`](https://huggingface.co/datasets/Jazhyc/gleipnir-kimi-k3-tool-trajectory-logits);
the training trajectories and benchmark rows are not redistributed here. The
training and evaluation code is in the
[Gleipnir repository](https://github.com/Jazhyc/gleipnir).

The adapter weights and release metadata are distributed under the MIT
License; see `LICENSE`. Users must also comply with the Qwen base-model terms,
the terms of the upstream datasets and any applicable teacher-provider terms.
Several upstream datasets declared no license when they were collected; the
public weights grant no right to reconstruct or redistribute them.

## Changelog

- **v1.0.** Card rewritten for the technical report; the weights are those
  of v0.2 (and v0.1). The repository was renamed from `Jazhyc/Gleipnir-27B`,
  which still redirects here. Recommends serving image 1.2.
- **v0.2 (2026-10-20).** Packaging release with the merged BF16 checkpoint and
  the optimized serving image; weights unchanged from v0.1.
- **v0.1 (2026-10-18).** First release.

## Citation

```bibtex
@techreport{ferrao2026gleipnir,
  title  = {Lightweight Restraints for Powerful Agents: AI Control Monitors That Decide in One Forward Pass},
  author = {Ferrao, Jeremias and Rajeev Kumar, Abhinav},
  year   = {2026},
  note   = {Technical report}
}
```

The arXiv identifier is added on publication. Pin the repository revision you
used in your experiments.
