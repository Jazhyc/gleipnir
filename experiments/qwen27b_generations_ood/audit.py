"""Offline interface audit: do Qwen3.6/3.8-27B render the frozen OOD prompts
exactly as Qwen3.5-27B does, within the engine limit and with the same
literal decision tokens? Tokenizer files only; no model weights or GPU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
    render_margin_prompt,
)
from gleipnir.binary_evaluation import binary_token_ids

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CONFIG = "experiments/tool_trajectory_monitoring/qwen27b_ood_benchmark.json"
MODELS = {
    "qwen35_27b": ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654"),
    "qwen36_27b": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"),
    "qwen38_27b": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
}


def audit() -> dict:
    from transformers import AutoTokenizer

    config = json.loads((ROOT / REFERENCE_CONFIG).read_text())
    prompt = config["prompt"]
    rows = [
        json.loads(line)
        for line in (ROOT / config["scope"]["input"]).read_text().splitlines()
    ]
    report = {"rows": len(rows), "models": {}}
    reference = None
    for name, (model_id, revision) in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        rendered = [
            render_margin_prompt(
                tokenizer,
                str(row["prompt"]),
                enable_thinking=prompt["enable_thinking"],
                assistant_suffix=prompt["assistant_suffix"],
                decision_prefix=prompt["decision_prefix"],
            )
            for row in rows
        ]
        lengths = [len(ids) for ids in tokenizer(rendered)["input_ids"]]
        digest = hashlib.sha256("\0".join(rendered).encode()).hexdigest()
        entry = {
            "model": model_id,
            "revision": revision,
            "decision_token_ids": binary_token_ids(tokenizer),
            "rendered_sha256": digest,
            "total_tokens": sum(lengths),
            "max_tokens": max(lengths),
            "rows_over_engine_limit": sum(
                n >= config["engine"]["max_model_len"] for n in lengths
            ),
        }
        if reference is None:
            reference = (rendered, lengths, entry)
            entry["matches_frozen_audit"] = (
                entry["total_tokens"] == config["engine"]["audited_total_prompt_tokens"]
                and entry["max_tokens"] == config["engine"]["audited_max_prompt_tokens"]
            )
        else:
            entry["rendered_rows_differing_from_qwen35"] = sum(
                a != b for a, b in zip(rendered, reference[0], strict=True)
            )
            entry["token_count_rows_differing_from_qwen35"] = sum(
                a != b for a, b in zip(lengths, reference[1], strict=True)
            )
            entry["same_decision_tokens"] = (
                entry["decision_token_ids"] == reference[2]["decision_token_ids"]
            )
        report["models"][name] = entry
        print(name, json.dumps(entry), flush=True)
    return report


if __name__ == "__main__":
    out = ROOT / "results/qwen27b_generations_ood/interface_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit(), indent=2) + "\n")
    print(out)
