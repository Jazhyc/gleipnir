"""Fresh FP32-master canary for the final QAD adapter, with BF16 arithmetic."""

import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.training_procedure_screen.evaluate_causal import score_adapter
from gleipnir.data.monitoring import file_hash, read_rows, write_json
from gleipnir.evaluation.binary import binary_token_ids


def main():
    root = Path.cwd()
    config = yaml.safe_load(
        (root / "experiments/quantization_recovery/config.yaml").read_text()
    )
    output = root / "results" / config["experiment"]
    complete = json.loads((output / "training_complete.json").read_text())
    master = output / "causal_adapter"
    if file_hash(master / "adapter_model.safetensors") != complete["master_sha256"]:
        raise ValueError("completed QAD master drift")
    old = json.loads(
        (
            root
            / "results/filtered-fp4-fa4-baseline01/4b/monitor/parity_reference.json"
        ).read_text()
    )
    by_id = {
        r["id"]: r
        for r in read_rows(root / "data" / config["experiment"] / "workload.jsonl")
    }
    rows = [by_id[i] for i in old["ids"]]
    tokenizer = AutoTokenizer.from_pretrained(root / config["base_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        root / config["base_model"],
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, master)
    scores, seconds = score_adapter(
        model,
        tokenizer,
        [r["input_ids"] for r in rows],
        binary_token_ids(tokenizer),
        batch_size=1,
        decision_head_mode="token_logits",
    )
    write_json(
        output / "parity_reference.json",
        {
            "ids": old["ids"],
            "base": old["base"],
            "adapter": scores,
            "master_sha256": complete["master_sha256"],
            "prompt_sha256": [r["prompt_sha256"] for r in rows],
            "seconds": seconds,
        },
    )


if __name__ == "__main__":
    main()
