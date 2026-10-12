"""Export producer records/raw Inspect archives and compare frozen OOD content."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from experiments.ood_visible_prose.restoration import calls
from experiments.ood_visible_prose.run import file_hash

PRODUCER = Path(
    "data/tool_trajectory_monitoring/source/blackboxmonitorsMATS-ccb77e3/control_arena.parquet"
)
PRODUCER_SHA = "17897773c5e7bceae54195482b2389888a30e191b8a5182380d32e1994d165f9"
LOG_DIRECTORY = Path("data/ood_visible_prose/original_logs")
LOG_HASHES = {
    "scheming-exploration__eval-logs__attack-tm.eval": (
        "e2c4f6a15af382065aa7d48d151c7c0d59fc0f94159f03e6b79889e8c6ff4488"
    ),
    "scheming-exploration__eval-logs__honest-tm.eval": (
        "f3aee32b009982b2d1f39d64cdc5a42cf7860252d67960aa017ac9d44f0592d0"
    ),
}
OUTPUT = Path("results/ood_visible_prose/original_control_comparison01")


def canonical(messages: list[dict[str, Any]], generic: bool, normalize: bool) -> list:
    """Preserve content/tool payloads; optionally remove empty formatting only."""
    output = []
    for message in messages:
        value = message.get("tool_calls") or []
        tools = (
            [{"function": t["name"], "arguments": t["arguments"]} for t in value]
            if generic
            else [{k: t[k] for k in ("function", "arguments")} for t in calls(value)]
        )
        text = message.get("content") or ""
        if not isinstance(text, str):
            raise ValueError("Comparison expects the producer's flattened strings")
        if normalize and message["role"] == "assistant":
            text = text.strip()
            if text.startswith("<think></think>"):
                text = text.removeprefix("<think></think>").strip()
            if not text and not tools:
                continue
        output.append((message["role"], text, tools))
    return output


def identity(value: Any) -> str:
    """Fingerprint canonical evidence without using labels to select a match."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def export_archive(path: Path, output: Path) -> dict[str, Any]:
    """Read native .eval sample JSON and retain messages/types without coercion."""
    counts: Counter[str] = Counter()
    source_hash = file_hash(path)
    with zipfile.ZipFile(path) as archive, output.open("w") as stream:
        header = json.loads(archive.read("header.json"))
        for member in sorted(archive.namelist()):
            if not member.startswith("samples/") or not member.endswith(".json"):
                continue
            sample = json.loads(archive.read(member))
            messages = sample["messages"]
            counts["samples"] += 1
            for message in messages:
                if message["role"] != "assistant":
                    continue
                content = message.get("content")
                counts["assistant_messages"] += 1
                if isinstance(content, list):
                    for block in content:
                        counts["block:" + block["type"]] += 1
                else:
                    counts["assistant_strings"] += 1
            stream.write(
                json.dumps(
                    {
                        "sample_id": sample["id"],
                        "epoch": sample["epoch"],
                        "messages": messages,
                        "metadata": sample.get("metadata"),
                        "scores": sample.get("scores"),
                        "source_member": member,
                        "source_sha256": source_hash,
                    }
                )
                + "\n"
            )
    return {
        "path": str(path),
        "sha256": file_hash(path),
        "counts": dict(counts),
        "eval_dataset": header["eval"]["dataset"],
        "export_sha256": file_hash(output),
        "cohort": "separate_bash_cohort_not_the_four_frozen_ood_sources",
    }


def main() -> None:
    """Compare the producer release and independently export actual raw logs."""
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    if file_hash(PRODUCER) != PRODUCER_SHA:
        raise ValueError("Producer release checksum mismatch")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = pq.read_table(PRODUCER).to_pylist()
    exact = defaultdict(list)
    normalized = defaultdict(list)
    tool_index = defaultdict(list)
    with (OUTPUT / "producer_export.jsonl").open("w") as stream:
        for record in records:
            messages = json.loads(record["messages"])
            a = canonical(messages, True, False)
            b = canonical(messages, True, True)
            item = {
                "id": record["id"],
                "messages": messages,
                "label": record["ground_truth"],
                "visibility": "flattened_unresolved",
            }
            exact[identity(a)].append(item)
            normalized[identity(b)].append(item)
            tool_index[
                identity([t for role, _, t in a if role == "assistant" and t])
            ].append(item)
            stream.write(json.dumps(item) + "\n")
    counts = {}
    with (OUTPUT / "matches.jsonl").open("w") as stream:
        for source in ("agentdojo", "iac", "rogue_eval", "shade_arena"):
            filename, checksum, expected_rows = config["sources"][source]
            path = Path(config["source_directory"]) / filename
            if file_hash(path) != checksum:
                raise ValueError("Frozen author source checksum mismatch")
            rows = pq.read_table(path).to_pylist()
            if len(rows) != expected_rows:
                raise ValueError("Frozen author source row mismatch")
            counter = Counter()
            for position, row in enumerate(rows):
                messages = json.loads(row["trajectory_data"])["messages"]
                a = canonical(messages, False, False)
                b = canonical(messages, False, True)
                precise = exact.get(identity(a), [])
                cleaned = normalized.get(identity(b), [])
                tool_matches = tool_index.get(
                    identity([t for role, _, t in a if role == "assistant" and t]), []
                )
                tools = [text for role, text, _ in a if role == "tool"]
                verified = [
                    candidate
                    for candidate in tool_matches
                    if [
                        m.get("content") or ""
                        for m in candidate["messages"]
                        if m["role"] == "tool"
                    ]
                    == tools
                ]
                counter["rows"] += 1
                counter["ordered_tool_payload_and_result_match"] += bool(verified)
                counter["exact_all_roles_content_match"] += bool(precise)
                counter["normalized_all_roles_content_match"] += bool(cleaned)
                counter["unique_normalized_match"] += len(cleaned) == 1
                counter["conflicting_labels_on_normalized_match"] += (
                    len({item["label"] for item in cleaned}) > 1
                )
                counter["label_agreement_after_match"] += any(
                    item["label"] == row["ground_truth"] for item in verified
                )
                stream.write(
                    json.dumps(
                        {
                            "source": source,
                            "source_position": position,
                            "author_id": row["id"],
                            "exact_producer_ids": [item["id"] for item in precise],
                            "normalized_producer_ids": [item["id"] for item in cleaned],
                            "tool_and_result_producer_ids": [
                                item["id"] for item in verified
                            ],
                        }
                    )
                    + "\n"
                )
            counts[source] = dict(counter)
    archives = []
    for filename, checksum in LOG_HASHES.items():
        path = LOG_DIRECTORY / filename
        if file_hash(path) != checksum:
            raise ValueError("Original raw archive checksum mismatch")
        archives.append(export_archive(path, OUTPUT / (filename + ".jsonl")))
    summary = {
        "producer_sha256": PRODUCER_SHA,
        "producer_repo": "Syghmon/blackboxmonitorsMATS",
        "producer_rows": len(records),
        "producer_revision": "ccb77e37589baaee1f0e16ac9ab89ceac8f6cc8a",
        "producer_license": "research-use-only",
        "counts": counts,
        "archives": archives,
        "normalization": (
            "leading_empty_think_wrapper_and_assistant_outer_whitespace_"
            "and_empty_callless_assistants_only"
        ),
        "restoration_accepted": False,
        "comparison_code_sha256": file_hash(Path(__file__)),
        "restoration_config_sha256": file_hash(Path(__file__).with_name("config.json")),
        "outputs": {p.name: file_hash(p) for p in OUTPUT.glob("*.jsonl")},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
