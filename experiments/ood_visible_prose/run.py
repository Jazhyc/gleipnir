"""Materialize only recoverable OOD public-prose inputs with a coverage receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from experiments.ood_visible_prose.restoration import (
    cua_skeleton,
    digest,
    restore,
    restore_cua,
    verify_provenance,
)
from experiments.tool_trajectory_monitoring.prompting import load_prompt_set


def file_hash(path: Path) -> str:
    """Hash an artifact without loading it into memory."""
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    """Prepare distinct corrected inputs or explicit unresolved-row receipts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--originals", type=Path)
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    frozen_path = Path(config["frozen_inputs"])
    if file_hash(frozen_path) != config["frozen_sha256"]:
        raise ValueError("Frozen input checksum mismatch")
    frozen = {}
    for line in frozen_path.open():
        row = json.loads(line)
        meta = row["metadata"]
        key = (meta["source_dataset"], meta["source_row_index"])
        if key in frozen or digest(row["prompt"]) != meta["rendered_prompt_sha256"]:
            raise ValueError("Duplicate frozen key or prompt mismatch")
        frozen[key] = row
    recovered = {}
    if args.originals:
        for line in args.originals.open():
            row = json.loads(line)
            key = (row["source"], row["row_index"])
            if key not in frozen or key in recovered:
                raise ValueError("Unknown/duplicate original trace key")
            if row["source"] not in {"agentdojo", "iac", "rogue_eval", "shade_arena"}:
                raise ValueError(
                    "Typed original overrides support ControlArena sources only"
                )
            verify_provenance(row)
            recovered[key] = row
    prompt_set = load_prompt_set()
    cua_source = config["cua_public_candidate"]
    cua_path = Path(cua_source["path"])
    if file_hash(cua_path) != cua_source["sha256"]:
        raise ValueError("CUA public candidate checksum mismatch")
    cua_candidates = {}
    for candidate in pq.read_table(cua_path).to_pylist():
        skeleton, _ = cua_skeleton(candidate["transcript"])
        identity = digest(json.dumps(skeleton))
        cua_candidates.setdefault(identity, []).append(candidate)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    source_receipts = []
    with (
        (output / "ready.jsonl.tmp").open("w") as ready,
        (output / "unresolved.jsonl.tmp").open("w") as unresolved,
    ):
        for source, (filename, expected_hash, expected_rows) in config[
            "sources"
        ].items():
            path = Path(config["source_directory"]) / filename
            if file_hash(path) != expected_hash:
                raise ValueError(f"Source checksum mismatch: {source}")
            table = pq.read_table(path)
            if table.num_rows != expected_rows:
                raise ValueError(f"Source count mismatch: {source}")
            source_receipts.append(
                {"source": source, "path": str(path), "sha256": expected_hash}
            )
            for source_row_index, row in enumerate(table.to_pylist()):
                key = (source, source_row_index)
                parent = frozen[key]
                meta = parent["metadata"]
                trajectory = row["transcript"]
                if (
                    prompt_set.student.render(trajectory) != parent["prompt"]
                    or row["ground_truth"] != meta["ground_truth"]
                    or row["row_index"] != meta["recorded_row_index"]
                ):
                    raise ValueError("Frozen source identity/label mismatch")
                counts[source + ":total"] += 1
                reason = None
                provenance = None
                if source in {"agentdojo", "iac", "rogue_eval", "shade_arena"}:
                    raw = json.loads(row["trajectory_data"])
                    if raw["id"] != row["id"]:
                        raise ValueError("Raw/source identity mismatch")
                    if key in recovered:
                        original = recovered[key]
                        trajectory = restore(
                            trajectory, raw["messages"], original["messages"]
                        )
                        provenance = original["provenance"]
                    else:
                        reason = "missing_original_typed_messages"
                elif source == "cua_shade_arena":
                    skeleton, _ = cua_skeleton(trajectory)
                    candidates = cua_candidates.get(digest(json.dumps(skeleton)), [])
                    if not candidates:
                        reason = "missing_exact_matching_original_public_transcript"
                    else:
                        versions = {
                            (c["transcript"], c["ground_truth"]) for c in candidates
                        }
                        if len(versions) != 1:
                            reason = "ambiguous_matching_cua_public_transcripts"
                        else:
                            candidate = candidates[0]
                            if candidate["ground_truth"] != row["ground_truth"]:
                                raise ValueError("Matched CUA label mismatch")
                            trajectory = restore_cua(
                                trajectory, candidate["transcript"]
                            )
                            provenance = cua_source
                if reason:
                    unresolved.write(
                        json.dumps(
                            {
                                "id": parent["id"],
                                "source": source,
                                "source_row_index": source_row_index,
                                "recorded_row_index": row["row_index"],
                                "parent_prompt_sha256": meta["rendered_prompt_sha256"],
                                "reason": reason,
                            }
                        )
                        + "\n"
                    )
                    counts[source + ":unresolved"] += 1
                    continue
                student = prompt_set.student.render(trajectory)
                teacher = prompt_set.teacher.render(trajectory)
                ready.write(
                    json.dumps(
                        {
                            "id": parent["id"],
                            "source": source,
                            "source_row_index": source_row_index,
                            "recorded_row_index": row["row_index"],
                            "label": row["ground_truth"],
                            "label_provenance": meta["ground_truth_provenance"],
                            "variant": config["variant"],
                            "trajectory": trajectory,
                            "trajectory_sha256": digest(trajectory),
                            "student_prompt": student,
                            "student_prompt_sha256": digest(student),
                            "teacher_prompt": teacher,
                            "teacher_prompt_sha256": digest(teacher),
                            "parent_prompt_sha256": meta["rendered_prompt_sha256"],
                            "changed": student != parent["prompt"],
                            "restoration_provenance": provenance,
                        }
                    )
                    + "\n"
                )
                counts[source + ":ready"] += 1
    total = sum(v for k, v in counts.items() if k.endswith(":total"))
    resolved = sum(v for k, v in counts.items() if k.endswith(":ready"))
    if total != len(frozen) or total != 6395:
        raise ValueError("Incomplete frozen coverage")
    for name in ("ready.jsonl", "unresolved.jsonl"):
        (output / (name + ".tmp")).replace(output / name)
    receipt = {
        "variant": config["variant"],
        "complete": resolved == total,
        "evaluation_ready": resolved == total,
        "frozen_rows": total,
        "ready_rows": resolved,
        "unresolved_rows": total - resolved,
        "counts": dict(counts),
        "parent_input_sha256": config["frozen_sha256"],
        "config_sha256": file_hash(Path(__file__).with_name("config.json")),
        "originals_sha256": file_hash(args.originals) if args.originals else None,
        "source_receipts": source_receipts,
        "cua_public_candidate": cua_source,
        "outputs": {
            name: file_hash(output / name)
            for name in ("ready.jsonl", "unresolved.jsonl")
        },
    }
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {k: receipt[k] for k in ("complete", "ready_rows", "unresolved_rows")}
        )
    )


if __name__ == "__main__":
    main()
