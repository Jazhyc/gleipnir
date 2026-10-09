"""Frozen task partitions and inserted-span provenance for the APPS replication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("config.json")


def partition(tasks: list[str], count: int, prefix: str) -> list[str]:
    """Select groups independently of labels, activations and scores."""
    if len(set(tasks)) != len(tasks) or not 0 < count < len(tasks):
        raise ValueError("invalid group partition")
    return sorted(
        tasks, key=lambda t: hashlib.sha256((prefix + t).encode()).hexdigest()
    )[:count]


def spans(prompt: str, template: str) -> list[tuple[int, int]]:
    """Locate the frozen transfer-template header and trailing docstring."""
    start = prompt.index("[assistant tool_call: submit]\n") + len(
        "[assistant tool_call: submit]\n"
    )
    end = prompt.rindex("</agent_trajectory>")
    header = 0 if template == "transfer_gpt4" else 84
    result = []
    if header:
        if prompt[start + header] != "\n":
            raise ValueError("template header drift")
        result.append((start, start + header + 1))
    doc_end = prompt.rindex('"""', start, end) + 3
    doc_start = prompt.rindex('\n"""', start, doc_end - 3)
    result.append((doc_start, doc_end))
    return result


def prepare() -> None:
    from tokenizers import Tokenizer

    config = json.loads(CONFIG.read_text())
    out = ROOT / "results/b200_injection_direction" / config["campaign_id"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "partition.json").exists():
        raise ValueError("prepared partition already exists")
    for name, spec in config["inputs"].items():
        if file_hash(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError("input drift " + name)
    all_rows = []
    for split in ["benchmark", "honest_controls"]:
        for r in read_rows(ROOT / config["inputs"][split]["path"]):
            if hashlib.sha256(r["prompt"].encode()).hexdigest() != r["prompt_sha256"]:
                raise ValueError("prompt hash drift")
            all_rows.append(r | {"population": split})
    if len(all_rows) != 9114 or len({r["id"] for r in all_rows}) != 9114:
        raise ValueError("APPS coverage drift")

    def task(r: dict) -> str:
        return str(r["metadata"]["task_id"])

    tasks = sorted({task(r) for r in all_rows})
    fit = set(partition(tasks, config["fit_tasks"], config["fit_hash_prefix"]))
    test = set(tasks) - fit
    adds = set(partition(sorted(test), config["add_tasks"], config["add_hash_prefix"]))
    parents = {
        (task(r), r["metadata"]["ground_truth"]): r
        for r in all_rows
        if r["metadata"]["condition"] in ["clean", "vanilla"]
    }
    tokenizer = Tokenizer.from_file(str(Path(config["model"]) / "tokenizer.json"))
    fitting = []
    for r in all_rows:
        if task(r) not in fit:
            continue
        cond = r["metadata"]["condition"]
        positions = []
        if cond.startswith(("inj_", "neutral_")):
            bounds = spans(r["prompt"], cond.split("_", 1)[1])
            stripped = r["prompt"]
            for a, b in reversed(bounds):
                stripped = stripped[:a] + stripped[b:]
            parent = parents[(task(r), r["metadata"]["ground_truth"])]["prompt"]
            marker = "[assistant tool_call: submit]\n"
            close = "</agent_trajectory>"
            a = stripped.index(marker) + len(marker)
            b = stripped.rindex(close)
            pa = parent.index(marker) + len(marker)
            pb = parent.rindex(close)
            if (
                stripped[a:b].rstrip("\n") != parent[pa:pb].rstrip("\n")
                or stripped[:a] != parent[:pa]
            ):
                raise ValueError("span removal does not recover parent " + r["id"])
            enc = tokenizer.encode(r["prompt"], add_special_tokens=False)
            if len(enc.ids) != r["prompt_tokens"]:
                raise ValueError("offset tokenizer count drift")
            positions = sorted(
                {
                    i
                    for a, b in bounds
                    for i, (s, e) in enumerate(enc.offsets)
                    if e > a and s < b
                }
            )
            if not positions:
                raise ValueError("missing inserted span")
        fitting.append(r | {"span_positions": positions})
    test_rows = [r for r in all_rows if task(r) in test]
    add_rows = [r for r in test_rows if task(r) in adds]
    id_rows = json.loads((ROOT / config["inputs"]["id"]["path"]).read_text())
    if len(fitting) != 896 or len(test_rows) != 8218 or len(id_rows) != 3012:
        raise ValueError("frozen population drift")
    for name, rows in [
        ("fit", fitting),
        ("apps", test_rows),
        ("add", add_rows),
        ("id", id_rows),
    ]:
        write_rows(out / (name + "_workload.jsonl"), rows)
    write_json(
        out / "partition.json",
        {
            "fit_tasks": sorted(fit),
            "test_tasks": sorted(test),
            "add_tasks": sorted(adds),
            "fit_rows": len(fitting),
            "test_rows": len(test_rows),
            "add_rows": len(add_rows),
            "id_rows": len(id_rows),
            "test_backdoor_tasks": sum(
                t in test and label == 1 for t, label in parents
            ),
            "add_backdoor_tasks": sum(t in adds and label == 1 for t, label in parents),
        },
    )
    write_json(out / "config.json", config)
    print(
        "injection_direction_prepared",
        len(fitting),
        len(test_rows),
        len(add_rows),
        flush=True,
    )


if __name__ == "__main__":
    prepare()
