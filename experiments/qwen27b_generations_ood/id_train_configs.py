"""Freeze Qwen3.8-27B configs for the ID and full training inputs.

Same interface and engine as the OOD run; only the scope and its offline
token audit (Qwen3.8 tokenizer, frozen rendering) change.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
    render_margin_prompt,
)
from gleipnir.qwen35_adapter_rebase import sha256_file

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INPUTS = {
    "id": "data/tool_trajectory_monitoring/teacher_id_benchmark",
    "train": "data/tool_trajectory_monitoring/teacher_training",
}


def main() -> None:
    from transformers import AutoTokenizer

    base = json.loads((HERE / "qwen38_27b.json").read_text())
    model = base["model"]
    tokenizer = AutoTokenizer.from_pretrained(model["id"], revision=model["revision"])
    prompt = base["prompt"]
    for split, directory in INPUTS.items():
        path = ROOT / directory / "prompts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rendered = [
            render_margin_prompt(
                tokenizer,
                row["prompt"],
                enable_thinking=prompt["enable_thinking"],
                assistant_suffix=prompt["assistant_suffix"],
                decision_prefix=prompt["decision_prefix"],
            )
            for row in rows
        ]
        lengths = [len(ids) for ids in tokenizer(rendered)["input_ids"]]
        config = copy.deepcopy(base)
        config["campaign_id"] = f"qwen27b-generations-{split}-qwen38_27b-v1"
        config["hypothesis"] = (
            "Descriptive: unadapted Qwen3.8-27B ranking on the "
            f"{split} rows under the frozen OOD interface, for comparison with "
            "Kimi K3 targets and trained students."
        )
        config["scope"] = {
            "role": f"descriptive_{split}_evaluation",
            "input": f"{directory}/prompts.jsonl",
            "input_sha256": sha256_file(path),
            "manifest": f"{directory}/prompts.manifest.json",
            "manifest_sha256": sha256_file(ROOT / directory / "prompts.manifest.json"),
            "rows": len(rows),
            "sources": sorted({r["metadata"]["source_dataset"] for r in rows}),
            "selection_rule": "Every row; nothing is tuned on these labels.",
        }
        config["engine"]["audited_total_prompt_tokens"] = sum(lengths)
        config["engine"]["audited_max_prompt_tokens"] = max(lengths)
        if max(lengths) >= config["engine"]["max_model_len"]:
            raise ValueError(f"{split} prompt exceeds the engine limit")
        out = HERE / f"qwen38_27b_{split}.json"
        out.write_text(json.dumps(config, indent=2) + "\n")
        print(split, len(rows), sum(lengths), max(lengths), out.name, flush=True)


if __name__ == "__main__":
    main()
