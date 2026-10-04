"""Freeze held-out synthetic ID views without rescoring the original adapter."""

from __future__ import annotations

import json
import random
import re
from collections import Counter

import yaml

from experiments.monitor_injection_augmentation.prepare import (
    DATA,
    HERE,
    ROOT,
    template_bank,
    verify_preparation,
)
from experiments.monitoring_hard_labels.prepare import validate_holdout
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from gleipnir.monitoring_campaign_data import (
    digest,
    file_hash,
    read_rows,
    trajectory_from_prompt,
    write_json,
    write_rows,
)
from gleipnir.transcript_injection import (
    ROLES,
    balanced_choices,
    draw_position,
    insert_message,
    remove_message,
)

EVAL_CONFIG = HERE / "eval_config.yaml"
EVAL_DATA = DATA / "evaluation"
TRANSFER = ROOT / "data/monitoring_injection_disentanglement"


def configuration() -> dict:
    return yaml.safe_load(EVAL_CONFIG.read_text())


def synthetic_views(rows: list[dict], bank: dict, seed: int) -> list[dict]:
    """Construct one reversible held-out-family view per original ID example."""
    rng = random.Random(seed)
    roles = balanced_choices(ROLES, len(rows), rng)
    templates = balanced_choices(
        [t for t in bank["templates"] if t["split"] == "heldout"], len(rows), rng
    )
    decisions = balanced_choices((0, 1), len(rows), rng)
    prompt_template = load_prompt_set().student
    result = []
    for row, role, template, decision in zip(
        rows, roles, templates, decisions, strict=True
    ):
        meta = row["metadata"]
        if digest(row["prompt"]) != meta["rendered_prompt_sha256"]:
            raise ValueError("source ID prompt drift")
        trajectory = trajectory_from_prompt(row["prompt"], meta["trajectory_sha256"])
        payload = template["text"].format(decision=decision)
        position = draw_position(rng)
        lowercase = re.compile(r"^\[(?:user|assistant|tool)\]\r?$", re.MULTILINE)
        if lowercase.match(trajectory):
            # A same-length structural view recognizes ID's lowercase headers.
            # Splice only the new block into the untouched original source bytes.
            structural = lowercase.sub(lambda m: m.group().upper(), trajectory)
            preview, receipt = insert_message(structural, role, payload, position)
            start = receipt["offset"]
            addition = preview[start : start + receipt["inserted_characters"]]
            addition = addition.replace(f"[{role.upper()}]\n", f"[{role}]\n", 1)
            augmented = trajectory[:start] + addition + trajectory[start:]
            receipt["format"] = "role_tagged_lowercase"
        else:
            augmented, receipt = insert_message(trajectory, role, payload, position)
        if remove_message(augmented, receipt) != trajectory:
            raise ValueError("ID source bytes changed")
        prompt = prompt_template.render(augmented)
        result.append(
            {
                "id": row["id"] + "/synthetic-heldout",
                "prompt": prompt,
                "metadata": {
                    **meta,
                    "base_id": row["id"],
                    "source_trajectory_sha256": meta["trajectory_sha256"],
                    "source_prompt_sha256": digest(row["prompt"]),
                    "trajectory_sha256": digest(augmented),
                    "rendered_prompt_sha256": digest(prompt),
                    "augmentation": {
                        **receipt,
                        "template_id": template["id"],
                        "template_family": template["family"],
                        "demanded_decision": decision,
                        "payload_sha256": digest(payload),
                        "label_provenance": (
                            "original_source; off_policy_invariance_assumption"
                        ),
                    },
                },
            }
        )
    if len({r["id"] for r in result}) != len(rows):
        raise ValueError("duplicate synthetic ID identity")
    return result


def verify_evaluation() -> dict:
    verify_preparation()
    manifest = json.loads((EVAL_DATA / "manifest.json").read_text())
    if manifest["eval_config_sha256"] != file_hash(EVAL_CONFIG):
        raise ValueError("evaluation configuration drift")
    for name, expected in manifest["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"evaluation source drift: {name}")
    for name, expected in manifest["files_sha256"].items():
        if file_hash(EVAL_DATA / name) != expected:
            raise ValueError("synthetic evaluation artifact drift")
    return manifest


def main() -> None:
    verify_preparation()
    if (EVAL_DATA / "manifest.json").exists():
        print(verify_evaluation()["summary"], flush=True)
        return
    config = configuration()
    sources = {}
    for key in ("baseline_summary", "baseline_reference"):
        path = ROOT / config[key]
        if file_hash(path) != config[key + "_sha256"]:
            raise ValueError("original baseline drift")
        sources[config[key]] = file_hash(path)
    baseline = json.loads((ROOT / config["baseline_summary"]).read_text())
    ident = baseline["identity"]
    for split in ("id", "canaries", "benchmark", "honest_controls"):
        path = TRANSFER / split / "neutral.jsonl"
        expected = (
            ident["id_input_sha256"]
            if split == "id"
            else ident["transfer_input_sha256"][split + "/neutral"]
        )
        if file_hash(path) != expected:
            raise ValueError("matched regular evaluation input drift")
        sources[str(path.relative_to(ROOT))] = expected
    rows = read_rows(TRANSFER / "id/neutral.jsonl")
    train = read_rows(
        ROOT / "data/student_injection_awareness/regular/student_rows.jsonl"
    )
    validate_holdout(train, rows)
    injected = synthetic_views(rows, template_bank(), config["seed"])
    validate_holdout(train, injected)
    for name in (
        "data/monitor_injection_augmentation/manifest.json",
        "experiments/monitor_injection_augmentation/prepare_eval.py",
        "experiments/judge_injection_continuation/evaluate.py",
    ):
        sources[name] = file_hash(ROOT / name)
    write_rows(EVAL_DATA / "synthetic_id.jsonl", injected)
    summary = {
        "rows": len(injected),
        "clean_pairs": len(rows),
        "roles": dict(Counter(r["metadata"]["augmentation"]["role"] for r in injected)),
        "templates": dict(
            Counter(r["metadata"]["augmentation"]["template_id"] for r in injected)
        ),
        "original_adapter_rescored": False,
        "training_lineage_overlap": 0,
    }
    write_json(
        EVAL_DATA / "manifest.json",
        {
            "eval_config_sha256": file_hash(EVAL_CONFIG),
            "source_sha256": sources,
            "files_sha256": {
                "synthetic_id.jsonl": file_hash(EVAL_DATA / "synthetic_id.jsonl")
            },
            "summary": summary,
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
