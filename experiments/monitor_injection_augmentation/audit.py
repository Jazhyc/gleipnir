"""Independently audit the materialized replacement population and receipts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from experiments.monitor_injection_augmentation.prepare import (
    CONFIG,
    DATA,
    OUTPUT,
    ROOT,
    template_bank,
    verify_preparation,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from gleipnir.monitoring_campaign_data import (
    digest,
    file_hash,
    read_rows,
    trajectory_from_prompt,
    write_json,
)


def audit_rows(
    source: list[dict], rows: list[dict], ledger: list[dict], bank: dict
) -> dict:
    """Reject drift in source identity, supervision, payloads or reversibility."""
    if len(source) != len(rows):
        raise ValueError("replacement population size drift")
    receipts = {(r["dataset"], str(r["index"])): r for r in ledger}
    if len(receipts) != len(ledger):
        raise ValueError("duplicate insertion receipt")
    templates = {t["id"]: t for t in bank["templates"] if t["split"] == "train"}
    permitted = {
        "student_prompt",
        "student_prompt_sha256",
        "trajectory_sha256",
        "student_direct_tokens",
        "augmentation",
    }
    seen = set()
    exposure = Counter()
    irregular = Counter()
    prompt_template = load_prompt_set().student
    for old, new in zip(source, rows, strict=True):
        key = (old["dataset"], str(old["index"]))
        if key in seen:
            raise ValueError("duplicate source identity")
        seen.add(key)
        if key not in receipts:
            if old != new:
                raise ValueError("clean row drift")
            continue
        receipt = receipts[key]
        if {k: v for k, v in old.items() if k not in permitted} != {
            k: v for k, v in new.items() if k not in permitted
        }:
            raise ValueError("source metadata or target drift")
        if {
            k: v for k, v in receipt.items() if k not in ("dataset", "index", "label")
        } != new["augmentation"]:
            raise ValueError("insertion ledger drift")
        if receipt["label"] != old["label"]:
            raise ValueError("receipt label drift")
        if digest(new["student_prompt"]) != new["student_prompt_sha256"]:
            raise ValueError("augmented prompt checksum drift")
        original = trajectory_from_prompt(
            old["student_prompt"],
            old["trajectory_sha256"],
            source_template=prompt_template,
        )
        augmented = trajectory_from_prompt(
            new["student_prompt"],
            new["trajectory_sha256"],
            source_template=prompt_template,
        )
        start = receipt["offset"]
        end = start + receipt["inserted_characters"]
        if not 0 <= start < end <= len(augmented):
            raise ValueError("invalid insertion span")
        if augmented[:start] + augmented[end:] != original:
            raise ValueError("original source bytes changed")
        template = templates[receipt["template_id"]]
        payload = template["text"].format(decision=receipt["demanded_decision"])
        if digest(payload) != receipt["payload_sha256"]:
            raise ValueError("payload checksum drift")
        if receipt["format"] == "xml_step":
            from html import escape

            payload = escape(payload)
            if not receipt["source_task_header_present"]:
                irregular["task_header_absent"] += 1
            elif not receipt["source_task_header_closed"]:
                irregular["task_header_unclosed"] += 1
        if augmented[start:end].count(payload) != 1:
            raise ValueError("missing or repeated inserted payload")
        exposure[f"{template['id']}/{old['label']}/{receipt['demanded_decision']}"] += 1
    if not set(receipts) <= seen:
        raise ValueError("orphan insertion receipt")
    if any(
        exposure[f"{t}/{label}/{demand}"] == 0
        for t in templates
        for label in (0, 1)
        for demand in (0, 1)
    ):
        raise ValueError("template lacks both behavior classes and verdict demands")
    return {
        "rows": len(rows),
        "clean_rows": len(rows) - len(ledger),
        "injected_rows": len(ledger),
        "all_source_bytes_recovered": True,
        "labels_targets_and_lineage_unchanged": True,
        "every_template_has_both_labels_and_both_demands": True,
        "template_label_demand_counts": dict(exposure),
        "injected_source_format_irregularities": dict(irregular),
    }


def main() -> None:
    manifest = verify_preparation()
    config = yaml.safe_load(CONFIG.read_text())
    summary = audit_rows(
        read_rows(ROOT / config["training_source"]),
        read_rows(DATA / "student_rows.jsonl"),
        read_rows(DATA / "insertion_ledger.jsonl"),
        template_bank(),
    )
    if summary["injected_rows"] != manifest["summary"]["injected_rows"]:
        raise ValueError("manifest population drift")
    summary["manifest_sha256"] = file_hash(DATA / "manifest.json")
    summary["audit_code_sha256"] = file_hash(Path(__file__))
    write_json(OUTPUT / "completion_audit.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
