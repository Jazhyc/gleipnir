"""Cache Gleipnir 4B v0.2's own BF16 scores on the 7,836 training rows.

Same-model QAD distils the BF16 model into its quantized self. The teacher is
the v0.2 merge built and parity-checked on this host, served by the stock BF16
monitor path (`quantization_recovery.run.server_command(bf16=True)`), over the
unchanged QAD training workload. Rows keep the format the training stage
checks: index, prompt hash and the two-class soft target.
"""

import asyncio
import json
import shutil
import subprocess
import time
from pathlib import Path

from experiments.qad_v02.generate import config as load_config
from experiments.quantization_recovery.run import retire, server_command, wait_server
from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows
from gleipnir.evaluation.http_score import score_batch


def main():
    root = Path.cwd()
    config = load_config(root)
    output = root / "results" / config["experiment"]
    directory = root / "data" / config["teacher_experiment"]
    directory.mkdir(parents=True, exist_ok=True)
    source = root / config["inputs"]["workload"]["path"]
    workload_path = directory / "workload.jsonl"
    if not workload_path.exists():
        shutil.copy2(source, workload_path)
    if file_hash(workload_path) != config["inputs"]["workload"]["sha256"]:
        raise ValueError("QAD training workload drift")
    rows = read_rows(workload_path)
    if (
        len(rows) != config["training_rows"]
        or sum(r["prompt_tokens"] for r in rows) != config["training_tokens"]
    ):
        raise ValueError("training population/token count drift")
    merge = json.loads((root / config["teacher_merge_artifact"]).read_text())
    for name, digest in merge["files_sha256"].items():
        if file_hash(Path(config["teacher_merge"]) / name) != digest:
            raise ValueError(f"v0.2 teacher merge drift: {name}")
    identity = {
        "teacher_master_sha256": config["initial_adapter_sha256"],
        "workload_sha256": file_hash(workload_path),
        "temperature": config["temperature"],
        "merged_files_sha256": merge["files_sha256"],
        "serving": "stock_bf16_monitor",
    }
    complete = directory / "teacher_complete.json"
    if complete.exists():
        receipt = json.loads(complete.read_text())
        if receipt["identity"] != identity or receipt["teacher_sha256"] != file_hash(
            directory / "teacher.jsonl"
        ):
            raise ValueError("completed teacher cache drift")
        return
    command, env = server_command(
        config, output, bf16=True, model=config["teacher_merge"]
    )
    write_json(
        output / "teacher_command.json", {"command": command, "identity": identity}
    )
    with (output / "teacher_server.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    async def score():
        await wait_server(process, config)
        targets = []
        for start in range(0, len(rows), 128):
            batch = rows[start : start + 128]
            values, _ = await score_batch(batch, config["evaluation"])
            for row, value in zip(batch, values, strict=True):
                if (
                    row["id"] != value["id"]
                    or row["prompt_sha256"] != value["prompt_sha256"]
                ):
                    raise ValueError("teacher row/prompt mismatch")
                targets.append(
                    {
                        "index": row["id"],
                        "dataset": row["dataset"],
                        "label": row["label"],
                        "soft_target": value["score"],
                        "target_probs": {
                            "negative": 1 - value["score"],
                            "positive": value["score"],
                        },
                        "raw_logits": value["logits"],
                        "rendered_prompt_sha256": row["prompt_sha256"],
                        "teacher_model": "Gleipnir-4B-v0.2_BF16",
                        "teacher_master_sha256": config["initial_adapter_sha256"],
                    }
                )
            print("teacher_rows", len(targets), len(rows), flush=True)
        write_rows(directory / "teacher.jsonl", targets)
        audit = json.loads((output / "bf16_audit.json").read_text())
        if not audit["passed"]:
            raise ValueError("BF16 teacher precision/dispatch audit failed")
        write_json(
            complete,
            {
                "rows": len(targets),
                "teacher_sha256": file_hash(directory / "teacher.jsonl"),
                "identity": identity,
                "temperature": config["temperature"],
                "finished_at_unix": time.time(),
            },
        )

    try:
        asyncio.run(score())
    finally:
        retire(process)


if __name__ == "__main__":
    main()
