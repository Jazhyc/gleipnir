"""Freeze the monitoring-anchored sampling schedule before GPU training."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import yaml

from gleipnir.binary_task_training import TaskMixtureSampler
from gleipnir.monitoring_campaign_data import (
    file_hash,
    read_rows,
    write_json,
    write_rows,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"
DATA = ROOT / "data/judge_injection_joint"
OUTPUT = ROOT / "results/judge_injection_joint"
PREFERENCES = ROOT / "data/judge_injection_continuation"


def configuration() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def sampler(rows: list[dict], config: dict) -> TaskMixtureSampler:
    return TaskMixtureSampler(
        [r["sampling_group"] for r in rows],
        [r["label"] for r in rows],
        seed=config["seed"],
        **config["task_mixture"],
    )


def verify_preparation() -> dict:
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["config_sha256"] != file_hash(CONFIG):
        raise ValueError("joint configuration drift")
    for name, expected in manifest["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"joint source drift: {name}")
    for name, expected in manifest["files_sha256"].items():
        if file_hash(DATA / name) != expected:
            raise ValueError(f"joint materialization drift: {name}")
    rows = read_rows(DATA / "student_rows.jsonl")
    if sampler(rows, configuration()).audit() != manifest["sampler"]:
        raise ValueError("joint sampler drift")
    return manifest


def main() -> None:
    if (DATA / "manifest.json").exists():
        print(json.dumps(verify_preparation()["sampler"]), flush=True)
        return
    config = configuration()
    sources = {}
    for key in (
        "monitor_source",
        "teacher_source",
        "preference_manifest",
        "startup_validation_reference",
    ):
        sources[config[key]] = file_hash(ROOT / config[key])
        if sources[config[key]] != config[key + "_sha256"]:
            raise ValueError(f"frozen upstream drift: {key}")
    pref_manifest = json.loads((ROOT / config["preference_manifest"]).read_text())
    for name in (
        "train.jsonl",
        "test.jsonl",
        "canary.jsonl",
        "split.json",
        "tokenizer/tokenizer.json",
        "tokenizer/chat_template.jinja",
    ):
        p = PREFERENCES / name
        if file_hash(p) != pref_manifest["files_sha256"][name]:
            raise ValueError("frozen preference input drift")
        sources[str(p.relative_to(ROOT))] = file_hash(p)
    from safetensors import safe_open

    fresh = ROOT / config["model"]["fresh_adapter"]
    with safe_open(str(fresh / "adapter_model.safetensors"), framework="pt") as handle:
        b_keys = [k for k in handle.keys() if "lora_B" in k]
        if len(b_keys) != 128 or any(
            handle.get_tensor(k).count_nonzero() for k in b_keys
        ):
            raise ValueError("initial adapter is not the fresh zero-B rank-128 layout")
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        sources[str((fresh / name).relative_to(ROOT))] = file_hash(fresh / name)
    monitors = read_rows(ROOT / config["monitor_source"])
    preferences = read_rows(PREFERENCES / "train.jsonl")
    test = read_rows(PREFERENCES / "test.jsonl")
    if (
        len(monitors) != 8688
        or len(preferences) != 16760
        or (
            {r["lineage_group"] for r in preferences}
            & {r["lineage_group"] for r in test}
        )
    ):
        raise ValueError("source coverage or query leakage")
    from gleipnir.monitoring_campaign_data import validate_targets

    validate_targets(monitors, read_rows(ROOT / config["teacher_source"]))
    rows = [
        {
            **r,
            "sampling_group": "monitor",
            "binary_objective": "soft",
            "decision_tokens": ["0", "1"],
            "decision_prefix": "Prediction:",
        }
        for r in monitors
    ]
    rows.extend(
        {
            **r,
            "dataset": "judge-preference/" + r["source"],
            "sampling_group": r["condition"],
            "binary_objective": "hard",
            "decision_tokens": ["A", "B"],
            "decision_prefix": "",
        }
        for r in preferences
    )
    draw = sampler(rows, config)
    schedule = draw.indices()
    write_rows(DATA / "student_rows.jsonl", rows)
    write_rows(
        DATA / "schedule.jsonl",
        [
            {
                "draw": n,
                "pool_index": i,
                "dataset": rows[i]["dataset"],
                "index": rows[i]["index"],
                "sampling_group": rows[i]["sampling_group"],
                "label": rows[i]["label"],
            }
            for n, i in enumerate(schedule)
        ],
    )
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(PREFERENCES / "tokenizer/tokenizer.json"))
    template = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    ).from_string((PREFERENCES / "tokenizer/chat_template.jinja").read_text())
    lengths = []
    for n, row in enumerate(rows, 1):
        prompt = template.render(
            messages=[{"role": "user", "content": row["student_prompt"]}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        lengths.append(
            len(
                tokenizer.encode(
                    prompt + row["decision_prefix"], add_special_tokens=False
                ).ids
            )
        )
        if n % 500 == 0:
            print(f"token_audit {n}/{len(rows)}", flush=True)
    if max(lengths) > 29696:
        raise ValueError("context exceeds validated envelope; no truncation")
    token_audit = {
        group: {
            "draws": sum(rows[i]["sampling_group"] == group for i in schedule),
            "input_tokens": sum(
                lengths[i] for i in schedule if rows[i]["sampling_group"] == group
            ),
            "maximum": max(
                lengths[i] for i in schedule if rows[i]["sampling_group"] == group
            ),
        }
        for group in sorted({r["sampling_group"] for r in rows})
    }
    manifest = {
        "campaign_id": config["campaign_id"],
        "config_sha256": file_hash(CONFIG),
        "source_sha256": sources,
        "monitoring_rows": len(monitors),
        "preference_pool_rows": len(preferences),
        "pool_rows": len(rows),
        "training_draws": len(schedule),
        "expected_steps": math.ceil(len(schedule) / 32),
        "sampler": draw.audit(),
        "tokens": token_audit,
        "selected_input_tokens": sum(lengths[i] for i in schedule),
        "context_maximum": max(lengths),
        "truncated": 0,
        "monitor_source_counts": dict(Counter(r["dataset"] for r in monitors)),
        "files_sha256": {
            p: file_hash(DATA / p) for p in ("student_rows.jsonl", "schedule.jsonl")
        },
    }
    write_json(DATA / "manifest.json", manifest)
    write_json(OUTPUT / "preparation.json", manifest)
    print(
        json.dumps({k: manifest[k] for k in ("sampler", "tokens", "expected_steps")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
