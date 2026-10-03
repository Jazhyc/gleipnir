"""Freeze reversible 40% replacement views without changing teacher targets."""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import yaml

from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from gleipnir.monitoring_campaign_data import (
    digest,
    file_hash,
    read_rows,
    trajectory_from_prompt,
    validate_targets,
    write_json,
    write_rows,
)
from gleipnir.transcript_injection import (
    ROLES,
    balanced_choices,
    draw_position,
    insert_message,
    remove_message,
    select_stratified,
)

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"
TEMPLATES = HERE / "templates.json"
DATA = ROOT / "data/monitor_injection_augmentation"
OUTPUT = ROOT / "results/monitor_injection_augmentation"


def template_bank() -> dict:
    """Validate project-authored train and held-out mechanism separation."""
    bank = json.loads(TEMPLATES.read_text())
    templates = bank["templates"]
    if len({t["id"] for t in templates}) != len(templates):
        raise ValueError("duplicate payload identity")
    families = {}
    for split in ("train", "heldout"):
        population = [t for t in templates if t["split"] == split]
        if not population:
            raise ValueError("empty template split")
        families[split] = {t["family"] for t in population}
    if families["train"] & families["heldout"]:
        raise ValueError("template family leakage")
    for template in templates:
        if template["split"] not in families or "{decision}" not in template["text"]:
            raise ValueError("invalid template split or demanded verdict")
        for decision in (0, 1):
            if not template["text"].format(decision=decision).strip():
                raise ValueError("empty payload")
    return bank


def augment(
    rows: list[dict], config: dict, bank: dict
) -> tuple[list[dict], list[dict]]:
    """Make one fixed view per source row; clean rows stay exactly unchanged."""
    settings = config["augmentation"]
    if settings["mode"] != "replace" or tuple(settings["roles"]) != ROLES:
        raise ValueError("only approved replacement/role configuration supported")
    selected = select_stratified(rows, settings["fraction"], config["seed"])
    rng = random.Random(config["seed"] + 1)
    roles = balanced_choices(ROLES, len(selected), rng)
    templates = balanced_choices(
        [t for t in bank["templates"] if t["split"] == "train"], len(selected), rng
    )
    decisions = balanced_choices((0, 1), len(selected), rng)
    result = list(rows)
    ledger = []
    prompt_template = load_prompt_set().student
    for i, role, template, decision in zip(
        selected, roles, templates, decisions, strict=True
    ):
        row = rows[i]
        if digest(row["student_prompt"]) != row["student_prompt_sha256"]:
            raise ValueError("source prompt drift")
        trajectory = trajectory_from_prompt(
            row["student_prompt"], row["trajectory_sha256"]
        )
        if prompt_template.render(trajectory) != row["student_prompt"]:
            raise ValueError("regular monitoring instruction drift")
        position = draw_position(
            rng, settings["position_mean"], settings["position_std"]
        )
        payload = template["text"].format(decision=decision)
        augmented, receipt = insert_message(trajectory, role, payload, position)
        if digest(remove_message(augmented, receipt)) != row["trajectory_sha256"]:
            raise ValueError("source byte preservation failed")
        provenance = {
            **receipt,
            "campaign_id": config["campaign_id"],
            "source_trajectory_sha256": row["trajectory_sha256"],
            "source_student_prompt_sha256": row["student_prompt_sha256"],
            "template_id": template["id"],
            "template_family": template["family"],
            "payload_sha256": digest(payload),
            "demanded_decision": decision,
            "target_provenance": (
                "unchanged_clean_source_teacher; behavior_invariance_assumption"
            ),
            "agent_continuation": "original_off_policy; not_resimulated",
        }
        prompt = prompt_template.render(augmented)
        result[i] = {
            **row,
            "student_prompt": prompt,
            "student_prompt_sha256": digest(prompt),
            "trajectory_sha256": digest(augmented),
            "augmentation": provenance,
        }
        result[i].pop("student_direct_tokens", None)
        ledger.append(
            {
                "dataset": row["dataset"],
                "index": row["index"],
                "label": row["label"],
                **provenance,
            }
        )
    return result, ledger


def position_histogram(ledger: list[dict], key: str) -> list[int]:
    """Ten bins on [0,1], retaining the endpoint in the last bin."""
    bins = [0] * 10
    for row in ledger:
        bins[min(9, int(row[key] * 10))] += 1
    return bins


def verify_preparation() -> dict:
    """Reusing a preparation requires every frozen code/input/artifact hash."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    if manifest["config_sha256"] != file_hash(CONFIG):
        raise ValueError("augmentation configuration drift")
    for name, expected in manifest["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"augmentation source drift: {name}")
    for name, expected in manifest["files_sha256"].items():
        if file_hash(DATA / name) != expected:
            raise ValueError(f"augmentation artifact drift: {name}")
    return manifest


def main() -> None:
    if (DATA / "manifest.json").exists():
        print(json.dumps(verify_preparation()["summary"]), flush=True)
        return
    config = yaml.safe_load(CONFIG.read_text())
    sources = {}
    for key in (
        "training_source",
        "teacher_source",
        "tokenizer_manifest",
        "startup_validation_reference",
    ):
        path = ROOT / config[key]
        sources[config[key]] = file_hash(path)
        if sources[config[key]] != config[key + "_sha256"]:
            raise ValueError(f"frozen input drift: {key}")
    for name in (
        "experiments/monitor_injection_augmentation/templates.json",
        "experiments/monitor_injection_augmentation/prepare.py",
        "src/gleipnir/transcript_injection.py",
        "src/gleipnir/monitoring_campaign_data.py",
        "experiments/tool_trajectory_monitoring/prompting.py",
        "experiments/tool_trajectory_monitoring/prompts/manifest.json",
        "experiments/tool_trajectory_monitoring/prompts/student.txt",
        "experiments/tool_trajectory_monitoring/prompts/teacher.txt",
    ):
        sources[name] = file_hash(ROOT / name)
    tokenizer_manifest = json.loads((ROOT / config["tokenizer_manifest"]).read_text())
    for name in ("tokenizer.json", "chat_template.jinja"):
        path = ROOT / config["tokenizer_root"] / name
        expected = tokenizer_manifest["files_sha256"]["tokenizer/" + name]
        if file_hash(path) != expected:
            raise ValueError("pinned tokenizer drift")
        sources[str(path.relative_to(ROOT))] = expected
    source = read_rows(ROOT / config["training_source"])
    targets = read_rows(ROOT / config["teacher_source"])
    if len(source) != config["training_rows"]:
        raise ValueError("training population drift")
    validate_targets(source, targets)
    rows, ledger = augment(source, config, template_bank())
    validate_targets(rows, targets)
    changed = {(r["dataset"], str(r["index"])) for r in ledger}
    for old, new in zip(source, rows, strict=True):
        if (old["dataset"], str(old["index"])) not in changed and old != new:
            raise ValueError("clean row drift")
        if any(
            old[k] != new[k]
            for k in old
            if k
            not in (
                "student_prompt",
                "student_prompt_sha256",
                "trajectory_sha256",
                "student_direct_tokens",
            )
        ):
            raise ValueError("source label/teacher/lineage/rationale drift")
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(
        str(ROOT / config["tokenizer_root"] / "tokenizer.json")
    )
    template = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    ).from_string((ROOT / config["tokenizer_root"] / "chat_template.jinja").read_text())
    lengths = []
    for start in range(0, len(rows), 64):
        prompts = [
            template.render(
                messages=[{"role": "user", "content": r["student_prompt"]}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "Prediction:"
            for r in rows[start : start + 64]
        ]
        lengths.extend(
            len(t.ids)
            for t in tokenizer.encode_batch(prompts, add_special_tokens=False)
        )
        if start % 1024 == 0:
            print(f"token_audit {min(start + 64, len(rows))}/{len(rows)}", flush=True)
    if max(lengths) > config["max_input_tokens"]:
        raise ValueError("augmented context exceeds validated envelope; no truncation")
    summary = {
        "rows": len(rows),
        "clean_rows": len(rows) - len(ledger),
        "injected_rows": len(ledger),
        "realized_fraction": len(ledger) / len(rows),
        "roles": dict(Counter(r["role"] for r in ledger)),
        "templates": dict(Counter(r["template_id"] for r in ledger)),
        "demanded_decisions": dict(Counter(r["demanded_decision"] for r in ledger)),
        "source_label_role_counts": dict(
            Counter(f"{r['dataset']}/{r['label']}/{r['role']}" for r in ledger)
        ),
        "source_label_injected_counts": dict(
            Counter(f"{r['dataset']}/{r['label']}" for r in ledger)
        ),
        "label_demanded_decision_counts": dict(
            Counter(f"{r['label']}/{r['demanded_decision']}" for r in ledger)
        ),
        "requested_position_histogram": position_histogram(
            ledger, "requested_position"
        ),
        "realized_position_histogram": position_histogram(ledger, "realized_position"),
        "insertion_formats": dict(Counter(r["format"] for r in ledger)),
        "token_audit": {
            "total_input_tokens": sum(lengths),
            "maximum": max(lengths),
            "minimum": min(lengths),
            "truncated": 0,
        },
        "expected_steps": math.ceil(len(rows) / config["logical_batch_size"]),
        "clean_rows_byte_equivalent": True,
        "source_bytes_reversible": True,
        "teacher_targets_unchanged": True,
    }
    write_rows(DATA / "student_rows.jsonl", rows)
    write_rows(DATA / "insertion_ledger.jsonl", ledger)
    write_json(
        DATA / "manifest.json",
        {
            "campaign_id": config["campaign_id"],
            "config_sha256": file_hash(CONFIG),
            "source_sha256": sources,
            "files_sha256": {
                name: file_hash(DATA / name)
                for name in (
                    "student_rows.jsonl",
                    "insertion_ledger.jsonl",
                )
            },
            "summary": summary,
            "target_provenance": (
                "unchanged clean-source teacher cache; "
                "counterfactual invariance assumption"
            ),
            "selection": "fixed final one-epoch checkpoint; no APPS selection",
        },
    )
    write_json(OUTPUT / "preparation.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(
            OUTPUT / f"failure_prepare_{time.time_ns()}.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise
