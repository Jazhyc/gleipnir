"""Freeze outcome-independent training membership and paired clean views."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
CONFIG = HERE / "config.json"


def key(row: dict) -> tuple[str, str]:
    return row["dataset"], str(row["index"])


def select(rows: list[dict], count: int, prefix: str) -> list[dict]:
    """Allocate proportional strata before deterministic hash selection."""
    if not 0 < count <= len(rows) or len({key(r) for r in rows}) != len(rows):
        raise ValueError("invalid sample")
    if len({r["lineage_group"] for r in rows}) != len(rows):
        raise ValueError("non-singleton lineage requires grouped selection")
    strata = defaultdict(list)
    for r in rows:
        strata[r["raw_source"], int(r["label"]), bool(r.get("augmentation"))].append(r)
    exact = {s: count * len(rs) / len(rows) for s, rs in strata.items()}
    quotas = {s: math.floor(v) for s, v in exact.items()}
    for s in sorted(strata, key=lambda s: (-(exact[s] - quotas[s]), s))[
        : count - sum(quotas.values())
    ]:
        quotas[s] += 1
    chosen = []
    for s, rs in sorted(strata.items()):
        chosen.extend(
            sorted(
                rs,
                key=lambda r: hashlib.sha256(
                    (prefix + ":".join(key(r))).encode()
                ).hexdigest(),
            )[: quotas[s]]
        )
    return sorted(chosen, key=key)


def prepare() -> None:
    from transformers import AutoTokenizer

    c = json.loads(CONFIG.read_text())
    out = ROOT / "results/b200_training_direction" / c["campaign_id"]
    for spec in c["inputs"].values():
        if file_hash(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError("input drift " + spec["path"])
    if out.exists():
        raise ValueError("output already prepared")
    original = {key(r): r for r in read_rows(ROOT / c["inputs"]["clean"]["path"])}
    teacher = {key(r): r for r in read_rows(ROOT / c["inputs"]["teacher"]["path"])}
    training = read_rows(ROOT / c["inputs"]["training"]["path"])
    if len(training) != 8688 or len(original) != 8688 or len(teacher) != 8688:
        raise ValueError("training coverage drift")
    chosen = select(training, c["sample_rows"], c["selection_prefix"])
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / c["tokenizer"], local_files_only=True
    )
    workload = []
    for r in chosen:
        old, t = original[key(r)], teacher[key(r)]
        if (
            r["label"] != old["label"]
            or t["rendered_prompt_sha256"] != r["teacher_rendered_prompt_sha256"]
        ):
            raise ValueError("label/teacher lineage drift")
        aug = r.get("augmentation", {})
        if aug and aug["source_student_prompt_sha256"] != old["student_prompt_sha256"]:
            raise ValueError("augmentation parent drift")
        views = [("actual", r)] + ([("clean_parent", old)] if aug else [])
        for view, v in views:
            if (
                hashlib.sha256(v["student_prompt"].encode()).hexdigest()
                != v["student_prompt_sha256"]
            ):
                raise ValueError("source prompt drift")
            prompt = (
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": v["student_prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                + "Prediction:"
            )
            tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if not 0 < tokens < 32768:
                raise ValueError("truncated training prompt")
            workload.append(
                {
                    "id": ":".join(key(r)) + ":" + view,
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt_tokens": tokens,
                    "metadata": {
                        "dataset": r["dataset"],
                        "index": str(r["index"]),
                        "source": r["raw_source"],
                        "label": int(r["label"]),
                        "lineage_group": r["lineage_group"],
                        "view": view,
                        "training_augmented": bool(aug),
                        "student_prompt_sha256": v["student_prompt_sha256"],
                        "soft_target": t["soft_target"],
                        "teacher_prompt_sha256": t["rendered_prompt_sha256"],
                        **{
                            k: aug.get(k)
                            for k in [
                                "template_id",
                                "template_family",
                                "demanded_decision",
                                "role",
                                "realized_position",
                            ]
                        },
                    },
                }
            )
    if set(
        r["metadata"]["template_id"] for r in workload if r["metadata"]["template_id"]
    ) != {r["augmentation"]["template_id"] for r in training if r.get("augmentation")}:
        raise ValueError("sample missing an augmentation template")
    out.mkdir(parents=True)
    write_rows(out / "workload.jsonl", workload)
    write_json(
        out / "prepare_receipt.json",
        {
            "selected_trajectories": len(chosen),
            "views": len(workload),
            "tokens_per_model": sum(r["prompt_tokens"] for r in workload),
            "maximum_tokens": max(r["prompt_tokens"] for r in workload),
            "strata": dict(
                Counter(
                    str((r["raw_source"], r["label"], bool(r.get("augmentation"))))
                    for r in chosen
                )
            ),
            "workload_sha256": file_hash(out / "workload.jsonl"),
            "selection": "proportional source/label/augmentation strata; fixed hash",
            "unique_original_lineage": True,
        },
    )
    print((out / "prepare_receipt.json").read_text(), flush=True)


if __name__ == "__main__":
    prepare()
