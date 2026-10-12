"""Score only the zero-adapter canary; this is not an intermediate teacher cache."""

import asyncio
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml
from safetensors import safe_open

from experiments.quantization_recovery.run import retire, server_command, wait_server
from gleipnir.data.monitoring import read_rows, write_json
from gleipnir.evaluation.http_score import score_batch
from gleipnir.training.distillation_targets import load_binary_targets


def main():
    root = Path.cwd()
    config = yaml.safe_load(
        (root / "experiments/kimi_direct_qat/config.yaml").read_text()
    )
    output = root / "results" / config["experiment"]
    workload = read_rows(
        root / "data" / config["teacher_experiment"] / "workload.jsonl"
    )
    _, binding = load_binary_targets(
        workload,
        root / config["teacher_cache"],
        expected_sha256=config["teacher_sha256"],
        teacher_model="moonshotai/kimi-k3",
        prompt_contract="privileged",
    )
    write_json(output / "initial_kimi_binding.json", binding)
    master = root / config["initial_adapter"] / "adapter_model.safetensors"
    with safe_open(master, framework="np") as archive:
        keys = [key for key in archive.keys() if "lora_B" in key]
        if not keys or any(np.any(archive.get_tensor(key) != 0) for key in keys):
            raise ValueError("fresh initialization must have zero LoRA B matrices")
    reference = json.loads(
        (
            root
            / "results/filtered-fp4-fa4-baseline01/4b/monitor/parity_reference.json"
        ).read_text()
    )
    by_id = {row["id"]: row for row in workload}
    canary = [by_id[identity] for identity in reference["ids"]]
    command, env = server_command(
        config, output, bf16=False, model=root / config["base_model"]
    )
    write_json(
        output / "initial_serving_command.json",
        {"command": command, "scope": "20_training_canary_prompts_unadapted_base"},
    )
    with (output / "initial_serving.log").open("x") as stream:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    async def score():
        await wait_server(process, config)
        predictions, seconds = await score_batch(
            canary, {**config["evaluation"], "concurrency": 4}
        )
        if len(predictions) != len(canary):
            raise ValueError("incomplete fresh-base canary")
        write_json(root / config["initial_server_canary"], predictions)
        write_json(
            output / "initial_serving_complete.json",
            {"seconds": seconds, "rows": len(predictions), "zero_B_verified": True},
        )

    try:
        asyncio.run(score())
    finally:
        retire(process)


if __name__ == "__main__":
    main()
