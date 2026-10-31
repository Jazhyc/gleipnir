"""Pre-registered cover-only smoke gate and pinned Amendment 1 restoration."""

from __future__ import annotations

import hashlib
import json

from experiments.adversarial_crossed_text.common import (
    ROOT,
    digest,
    order_key,
    read_jsonl,
    sha,
    write_json,
)

RETRYABLE = {
    "length_out_of_band",
    "banned_wording",
    "empty",
    "code_block",
    "truncation",
}


def row_hash(row):
    return sha(json.dumps(row, sort_keys=True))


def smoke_tasks(out, cfg, selection, allowed):
    spec = cfg["amendment_3"]["smoke"]
    tasks = sorted(
        (t for t in selection["development"] if t in allowed),
        key=lambda t: (order_key(spec["hash_seed"], t), t),
    )[: spec["tasks"]]
    if len(tasks) != spec["tasks"]:
        raise SystemExit("insufficient eligible development tasks for smoke gate")
    if spec.get("selected_tasks", tasks) != tasks:
        raise ValueError("smoke tasks differ from preregistered IDs")
    record = {"tasks": tasks, "rule": spec["rule"], "seed": spec["hash_seed"]}
    path = out / "smoke.tasks.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("smoke selection changed")
    write_json(path, record)
    return tasks


def original_history(out, name):
    """Read pre-restoration history even if an append stopped mid-JSON-line."""
    manifest = out / "fallback.manifest.json"
    if manifest.exists():
        segment = json.loads(manifest.read_text())["segments"].get(name)
        if segment:
            data = (out / name).read_bytes()[: segment["offset"]]
            return [
                json.loads(line) for line in data.decode().splitlines() if line.strip()
            ]
    return read_jsonl(out / name)


def require_current_evidence(out, evidence):
    rows = {
        name: {key(r): r for r in read_jsonl(out / name)}
        for name, key in (
            ("texts.jsonl", lambda r: (r["task_id"], r["condition"])),
            ("hand_audit.jsonl", lambda r: (r["task_id"], r["condition"])),
            ("calls.jsonl", lambda r: r["request_id"]),
        )
    }
    for e in evidence:
        key = e["task_id"], "cover_story"
        text = rows["texts.jsonl"][key]
        receipt = rows["calls.jsonl"][f"text:{key[0]}:cover_story:{text['attempt']}"]
        if any(
            row_hash(row) != e[field]
            for row, field in (
                (text, "text_row_sha256"),
                (rows["hand_audit.jsonl"][key], "hand_row_sha256"),
                (receipt, "receipt_row_sha256"),
            )
        ):
            raise ValueError("smoke evidence superseded; continuation forbidden")


def decision_from_evidence(out, cfg, evidence):
    """Re-derive a decision from immutable history, including after restoration."""
    from experiments.adversarial_crossed_text.pipeline import hand_verdict

    histories = {
        name: {row_hash(r): r for r in original_history(out, name)}
        for name in ("texts.jsonl", "hand_audit.jsonl", "calls.jsonl")
    }
    tasks = json.loads((out / "smoke.tasks.json").read_text())["tasks"]
    if [e["task_id"] for e in evidence] != tasks:
        raise ValueError("smoke decision tasks changed")
    finished = valid = 0
    for e in evidence:
        try:
            text = histories["texts.jsonl"][e["text_row_sha256"]]
            hand = histories["hand_audit.jsonl"][e["hand_row_sha256"]]
            receipt = histories["calls.jsonl"][e["receipt_row_sha256"]]
        except KeyError:
            raise ValueError("smoke evidence missing or changed") from None
        task = e["task_id"]
        if (
            text["task_id"] != task
            or text["condition"] != "cover_story"
            or text["text_sha256"] != sha(text["text"])
            or hand["task_id"] != task
            or hand["condition"] != "cover_story"
            or hand["text_sha256"] != text["text_sha256"]
            or hand["text_status"] != text["status"]
            or receipt["request_id"] != f"text:{task}:cover_story:{text['attempt']}"
        ):
            raise ValueError("smoke evidence identity mismatch")
        if receipt["status"] == "reserved" and text["status"] == "interrupted_unknown":
            result = {"status": "interrupted_unknown"}
        elif receipt["status"] == "settled" and "result" in receipt:
            result = receipt["result"]
        else:
            raise ValueError(
                "smoke receipt is neither settled nor terminal interrupted"
            )
        complete = (
            result["status"] == "ok"
            and result.get("finish_reason") == "stop"
            and bool(text["text"].strip())
        )
        if (
            text["status"] in RETRYABLE
            and text["attempt"] < cfg["amendment_2"]["max_attempts"]
        ):
            raise SystemExit(
                "finish both permitted format attempts before deciding smoke"
            )
        finished += complete
        valid += (
            complete
            and text["status"] == "ok"
            and hand_verdict(
                hand, {k: v for k, v in hand.items() if k not in {"valid", "reason"}}
            )
        )
    spec = cfg["amendment_3"]["smoke"]
    return {
        "evidence": evidence,
        "finished": finished,
        "hand_valid": valid,
        "branch": "proceed"
        if finished >= spec["minimum_finished"] and valid >= spec["minimum_hand_valid"]
        else "fallback",
        "basis": "text validity only; no monitor scores exist",
    }


def smoke_gate(out, cfg, pairs=None, triggers=None):
    from experiments.adversarial_crossed_text.pipeline import (
        cell_identity,
        final_texts,
        hand_verdict,
        text_cells,
    )

    path = out / "smoke.gate.json"
    if path.exists():
        record = json.loads(path.read_text())
        if decision_from_evidence(out, cfg, record["evidence"]) != record:
            raise ValueError("smoke decision changed")
        if record["branch"] == "proceed":
            require_current_evidence(out, record["evidence"])
        return record
    selection = json.loads((out / "selection.json").read_text())
    tasks = json.loads((out / "smoke.tasks.json").read_text())["tasks"]
    cells = text_cells(selection, final_texts(out), {}, cfg)
    hands = {
        (r["task_id"], r["condition"]): r for r in read_jsonl(out / "hand_audit.jsonl")
    }
    calls = {r["request_id"]: r for r in read_jsonl(out / "calls.jsonl")}
    evidence = []
    for task in tasks:
        cell = cells[task, "cover_story"]
        text = cell["record"]
        hand = hands.get((task, "cover_story"))
        if not text or not hand:
            raise SystemExit("hand-read all five completed smoke cells before deciding")
        hand_verdict(hand, cell_identity(cell, pairs[task], triggers[task]))
        receipt = calls[f"text:{task}:cover_story:{text['attempt']}"]
        evidence.append(
            {
                "task_id": task,
                "text_row_sha256": row_hash(text),
                "hand_row_sha256": row_hash(hand),
                "receipt_row_sha256": row_hash(receipt),
            }
        )
    record = decision_from_evidence(out, cfg, evidence)
    write_json(path, record)
    return record


def fallback_payloads(cfg):
    """Restore the source final cover collection; its 34 valid verdicts are unchanged."""
    spec = cfg["amendment_3"]["fallback_import"]
    source = ROOT / spec["source"]
    for name, expected in spec["files"].items():
        if digest(source / name) != expected:
            raise ValueError(f"fallback source artifact changed: {name}")
    identity = json.loads((source / "pipeline.identity.json").read_text())
    if identity["config_sha256"] != spec["config_sha256"] or identity["dry_run"]:
        raise ValueError("fallback source configuration or mode changed")
    texts = {}
    for row in read_jsonl(source / "texts.jsonl"):
        if row["condition"] == "cover_story":
            if sha(row["text"]) != row["text_sha256"]:
                raise ValueError("fallback text hash mismatch")
            texts[row["task_id"]] = row
    rows = {
        "texts.jsonl": list(texts.values()),
        "text_checks.jsonl": [
            r
            for r in read_jsonl(source / "text_checks.jsonl")
            if r["condition"] == "cover_story"
        ],
        "hand_audit.jsonl": [
            r
            for r in read_jsonl(source / "hand_audit.jsonl")
            if r["condition"] == "cover_story"
        ],
    }
    return {
        n: "".join(json.dumps(r) + "\n" for r in rs).encode() for n, rs in rows.items()
    }


def restore_fallback(out, cfg):
    if smoke_gate(out, cfg)["branch"] != "fallback":
        raise SystemExit("fallback requires a failed text-validity smoke gate")
    payloads = fallback_payloads(cfg)
    path = out / "fallback.manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
    else:
        manifest = {
            "source": cfg["amendment_3"]["fallback_import"],
            "segments": {
                n: {
                    "offset": (out / n).stat().st_size if (out / n).exists() else 0,
                    "prefix_sha256": hashlib.sha256(
                        (out / n).read_bytes() if (out / n).exists() else b""
                    ).hexdigest(),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                for n, data in payloads.items()
            },
        }
        write_json(path, manifest)
    if manifest["source"] != cfg["amendment_3"]["fallback_import"]:
        raise ValueError("fallback provenance changed")
    for name, data in payloads.items():
        segment = manifest["segments"][name]
        if (
            segment["bytes"] != len(data)
            or segment["sha256"] != hashlib.sha256(data).hexdigest()
        ):
            raise ValueError("fallback segment identity changed")
        target = out / name
        current = target.read_bytes() if target.exists() else b""
        if (
            hashlib.sha256(current[: segment["offset"]]).hexdigest()
            != segment["prefix_sha256"]
        ):
            raise ValueError("fallback original history changed")
        suffix = current[segment["offset"] :]
        if len(current) < segment["offset"] or suffix != data[: len(suffix)]:
            raise ValueError("fallback segment changed")
        with target.open("ab") as handle:
            handle.write(data[len(suffix) :])
    validate_fallback(out, cfg)


def validate_fallback(out, cfg):
    path = out / "fallback.manifest.json"
    if not path.exists():
        return
    if smoke_gate(out, cfg)["branch"] != "fallback":
        raise ValueError("fallback contradicts smoke decision")
    manifest = json.loads(path.read_text())
    if manifest["source"] != cfg["amendment_3"]["fallback_import"]:
        raise ValueError("fallback provenance changed")
    for name, data in fallback_payloads(cfg).items():
        segment = manifest["segments"][name]
        current = (out / name).read_bytes()
        if (
            hashlib.sha256(current[: segment["offset"]]).hexdigest()
            != segment["prefix_sha256"]
        ):
            raise ValueError("fallback original history changed")
        actual = current[segment["offset"] : segment["offset"] + len(data)]
        suffix = current[segment["offset"] + len(data) :]
        if any(
            json.loads(line).get("condition") == "cover_story"
            for line in suffix.decode().splitlines()
            if line.strip()
        ):
            raise ValueError("restored fallback cover collection overridden")
        if (
            segment["bytes"] != len(data)
            or segment["sha256"] != hashlib.sha256(data).hexdigest()
            or actual != data
        ):
            raise ValueError("restored fallback artifact changed")
