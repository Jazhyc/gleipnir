"""Provenance-preserving JudgeDeceiver preference pairs and grouped augmentation."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalized(value: str) -> str:
    return " ".join(value.split())


def candidate_outputs(row: dict[str, Any]) -> dict[int, str]:
    outputs = {
        int(key.removeprefix("output_")): value
        for key, value in row.items()
        if key.startswith("output_")
    }
    if sorted(outputs) != list(range(1, len(outputs) + 1)) or any(
        not isinstance(value, str) or not value.strip() for value in outputs.values()
    ):
        raise ValueError("invalid candidate outputs")
    return outputs


def case_preferences(
    family: str, row: dict[str, Any], attacked_answer: str
) -> list[tuple[str, str]]:
    """Derive only supported preferences; never rank two unlabeled search entries."""
    outputs = candidate_outputs(row)
    attacked = int(row["label"])
    if attacked not in outputs or normalized(outputs[attacked]) != normalized(
        attacked_answer
    ):
        raise ValueError("attacker label does not identify its released target")
    if family == "metatool":
        preferred = int(row["real_label"])
        if preferred not in outputs or preferred == attacked:
            raise ValueError("invalid true tool label")
        return [(outputs[preferred], v) for k, v in outputs.items() if k != preferred]
    if family == "rlaif" and len(outputs) != 2:
        raise ValueError("RLAIF must have two candidates")
    if family not in {"search", "rlaif"}:
        raise ValueError("unknown case-study label contract")
    return [(v, outputs[attacked]) for k, v in outputs.items() if k != attacked]


def load_pairs(raw: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Use every source row, merge duplicate pairs and union applicable payloads."""
    pairs: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, int] = defaultdict(int)

    def add(
        family: str,
        question: str,
        preferred: str,
        other: str,
        payloads: dict[str, str],
        provenance: dict[str, Any],
    ) -> None:
        if normalized(preferred) == normalized(other):
            raise ValueError("preference candidates are identical")
        group = digest(normalized(question))
        identity = digest(json.dumps([family, group, preferred, other]))
        pair = pairs.setdefault(
            identity,
            {
                "pair_id": identity,
                "source": family,
                "group": group,
                "question": question,
                "preferred": preferred,
                "other": other,
                "payloads": {},
                "provenance": [],
            },
        )
        pair["provenance"].append(provenance)
        for name, payload in payloads.items():
            if not payload.strip():
                raise ValueError("empty attack payload")
            key = digest(payload)
            record = pair["payloads"].setdefault(key, {"text": payload, "names": []})
            if name not in record["names"]:
                record["names"].append(name)

    for family in ("llmbar", "mtbench"):
        targets = {
            r["question_id"]: r
            for r in json.loads(
                (raw / f"dataset/results_suffix/basic/{family}.json").read_text()
            )
        }
        for split in ("data_for_train", "data_for_eval"):
            for path in sorted((raw / f"dataset/{split}/basic/{family}").glob("*.csv")):
                question_id = int(path.stem.rsplit("_", 1)[1])
                target = targets[question_id]
                with path.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                for index, row in enumerate(rows):
                    source_counts[family] += 1
                    if row["target"] not in {
                        "Output (a) is better.",
                        "Output (b) is better.",
                    }:
                        raise ValueError("unknown main-data attacker label")
                    attacked = "text1" if "(a)" in row["target"] else "text2"
                    other_key = "text2" if attacked == "text1" else "text1"
                    if normalized(row[attacked]) != normalized(target["bad_answer"]):
                        raise ValueError("main-data attack answer mismatch")
                    _, separator, question = row["instruction"].partition(
                        "# Instruction: "
                    )
                    if not separator or normalized(question) != normalized(
                        target["question"]
                    ):
                        raise ValueError("main-data question mismatch")
                    add(
                        family,
                        target["question"],
                        row[other_key],
                        row[attacked],
                        {
                            f"{family}/{question_id}/{m}": target[m]
                            for m in ("mistral", "openchat_3.5", "llama-2", "llama-3")
                        },
                        {
                            "file": str(path.relative_to(raw)),
                            "row": index,
                            "attacker_label": row["target"],
                            "label_provenance": "constructed_clean_vs_bad_answer",
                        },
                    )
    for family, id_key, text_key in (
        ("rlaif", "input_id", "bad_answer"),
        ("search", "input_id", "bad_entry"),
        ("metatool", "tool_id", "bad_tool"),
    ):
        targets = {
            int(r[id_key]): r
            for r in json.loads(
                (raw / f"dataset/results_suffix/case_study/{family}.json").read_text()
            )
        }
        for path in sorted(
            (raw / f"dataset/data_for_eval/case_study/{family}").glob("*.json")
        ):
            question_id = int(path.stem.split("_")[1])
            target = targets[question_id]
            for index, row in enumerate(json.loads(path.read_text())):
                source_counts[family] += 1
                for preferred, other in case_preferences(family, row, target[text_key]):
                    add(
                        family,
                        row["input"],
                        preferred,
                        other,
                        {f"{family}/{question_id}": target["suffix"]},
                        {
                            "file": str(path.relative_to(raw)),
                            "row": index,
                            "attacker_label": row["label"],
                            "real_label": row.get("real_label"),
                            "label_provenance": "explicit_real_label"
                            if family == "metatool"
                            else "constructed_clean_vs_attack_target",
                        },
                    )
    expected = {
        "llmbar": 1120,
        "mtbench": 1120,
        "rlaif": 100,
        "search": 300,
        "metatool": 150,
    }
    if dict(source_counts) != expected:
        raise ValueError(f"source coverage drift: {dict(source_counts)}")
    return sorted(pairs.values(), key=lambda r: r["pair_id"]), dict(source_counts)


def grouped_split(pairs: list[dict[str, Any]], seed: int = 0) -> dict[str, str]:
    """Choose a near-20% row holdout while preserving entire source queries."""
    groups: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pair in pairs:
        groups[pair["source"]][pair["group"]] += 2 * (1 + 2 * len(pair["payloads"]))
    if len({g for values in groups.values() for g in values}) != sum(
        map(len, groups.values())
    ):
        raise ValueError(
            "cross-source query overlap requires a global split allocation"
        )
    selections = []
    for source in sorted(groups):
        values = groups[source]
        count = {"llmbar": 2, "mtbench": 2, "rlaif": 1, "search": 1, "metatool": 0}[
            source
        ]
        ordered = sorted(values, key=lambda g: digest(f"{seed}:{g}"))
        selections.append(list(itertools.combinations(ordered, count)))
    weights = {g: n for values in groups.values() for g, n in values.items()}
    total = sum(weights.values())
    choices = (
        tuple(g for subset in selection for g in subset)
        for selection in itertools.product(*selections)
    )
    held_out = min(
        choices,
        key=lambda choice: (
            abs(5 * sum(weights[g] for g in choice) - total),
            digest(f"{seed}:" + ":".join(sorted(choice))),
        ),
    )
    return {g: "test" if g in held_out else "train" for g in weights}


def augment_pairs(
    pairs: list[dict[str, Any]], split: dict[str, str], instruction: str
) -> list[dict[str, Any]]:
    """Keep decision labels independent of injection presence and answer order."""
    rows = []
    for pair in pairs:
        variants = [("clean", None, "")]
        for payload_id, payload in sorted(pair["payloads"].items()):
            variants.extend(
                [
                    ("preferred_injected", payload_id, payload["text"]),
                    ("disfavored_injected", payload_id, payload["text"]),
                ]
            )
        for order in (0, 1):
            for condition, payload_id, suffix in variants:
                preferred, other = pair["preferred"], pair["other"]
                if condition == "preferred_injected":
                    preferred += " " + suffix
                elif condition == "disfavored_injected":
                    other += " " + suffix
                a, b = (preferred, other) if order == 0 else (other, preferred)
                prompt = (
                    instruction.strip()
                    + "\n\nUser request:\n"
                    + pair["question"]
                    + "\n\n<candidate_A>\n"
                    + a
                    + "\n</candidate_A>"
                    + "\n\n<candidate_B>\n"
                    + b
                    + "\n</candidate_B>"
                    + "\n\nSelect A or B:"
                )
                identity = digest(f"{pair['pair_id']}:{order}:{condition}:{payload_id}")
                label = order
                rows.append(
                    {
                        "index": identity,
                        "id": identity,
                        "dataset": "judgedeceiver/" + pair["source"],
                        "source": pair["source"],
                        "label": label,
                        "label_match": True,
                        "student_target": "AB"[label],
                        "student_prompt": prompt,
                        "student_prompt_sha256": digest(prompt),
                        "pair_id": pair["pair_id"],
                        "lineage_group": pair["group"],
                        "split": split[pair["group"]],
                        "condition": condition,
                        "payload_id": payload_id,
                        "order": order,
                        "parse_error": False,
                    }
                )
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate augmented identity")
    by_split = {
        side: {r["lineage_group"] for r in rows if r["split"] == side}
        for side in ("train", "test")
    }
    if by_split["train"] & by_split["test"]:
        raise ValueError("grouped holdout leakage")
    return rows
