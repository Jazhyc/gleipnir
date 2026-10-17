"""Export the seed-0 Qwen3.8-27B initial LoRA adapter for campaigns.

Rebuilds the trainer's fresh LoRA (``adapter_init_seed=0``: seed, then
``get_peft_model`` with the distillation config's rank-128 targets), casts the
trainable tensors to the FP32 master dtype, and writes a PEFT adapter only if
its tensor digest equals the ``initial_master_sha256`` the 27B benchmark
recorded for the same initialization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from experiments.qwen38_27b_training_speed.prepare import MODEL, REVISION
from gleipnir.data.monitoring import file_hash
from gleipnir.training.execution_audit import tensor_digest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = (
    ROOT / "results/qwen38_27b_training_speed/screen_control01/bf16/packed/screen.json"
)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EXCLUDE = ".*(visual|vision_tower|merger|patch_embed).*"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(BENCHMARK.read_text())["initial_master_sha256"]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": torch.cuda.current_device()},
    )
    config = dict(
        r=128,
        lora_alpha=256,
        lora_dropout=0.0,
        target_modules=TARGETS,
        exclude_modules=EXCLUDE,
        task_type="CAUSAL_LM",
    )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = get_peft_model(model, LoraConfig(**config))
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    master = {n: p.detach().float().cpu() for n, p in named}
    digest = tensor_digest(list(master.values()))
    print(f"digest={digest} expected={expected} tensors={len(master)}", flush=True)
    if digest != expected:
        raise SystemExit("initialization does not reproduce the benchmark digest")
    if any(t.abs().sum() != 0 for n, t in master.items() if ".lora_B." in n):
        raise SystemExit("LoRA B must start at zero")
    args.output.mkdir(parents=True, exist_ok=False)
    LoraConfig(
        **config, base_model_name_or_path=MODEL, revision=REVISION
    ).save_pretrained(args.output)
    # PEFT adapter keys omit the adapter name, as in the 4B initial adapter.
    weights = {n.replace(".default.", "."): t.contiguous() for n, t in master.items()}
    save_file(
        weights, args.output / "adapter_model.safetensors", metadata={"format": "pt"}
    )
    receipt = {
        "derivation": "Seed-0 PEFT LoRA as built by the trainer (adapter_init_seed=0); "
        "A from PEFT's default init, B zero; digest equals the 27B benchmark's.",
        "initial_master_sha256": digest,
        "benchmark": str(BENCHMARK.relative_to(ROOT)),
        "trainable_tensors": len(master),
        "files_sha256": {p.name: file_hash(p) for p in sorted(args.output.iterdir())},
    }
    (args.output / "initialization_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
