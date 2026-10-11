"""Freeze literal filtering rules and five launchable monitoring campaigns."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from gleipnir.data.monitoring import (
    digest,
    file_hash,
    read_rows,
    validate_targets,
    write_json,
)
from gleipnir.data.teacher_filter import remove_disagreement

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/teacher_disagreement_filtering/sweep04"
CASES = {
    "control": None,
    "delta000": 0.0,
    "delta005": 0.05,
    "delta010": 0.1,
    "delta020": 0.2,
}
EXPECTED = {
    "control": 0,
    "delta000": 852,
    "delta005": 796,
    "delta010": 723,
    "delta020": 627,
}
CENSUS = "results/b200_training_clean_direction/clean_direction01/paired_models.jsonl"
CENSUS_SHA = "ebf8fa4cd24b49a9b164ea77931e539cba3e7949c7b0db4fd3a02541571cb650"


def identity(row: dict) -> tuple[str, str]:
    """Use the canonical student/teacher identity, never a positional join."""
    return row["dataset"], str(row["index"])


def subset(source: Path, target: Path, excluded: set[tuple[str, str]]) -> None:
    """Preserve exact retained JSONL bytes and order."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as inp, target.open("xb") as out:
        for line in inp:
            if identity(json.loads(line)) not in excluded:
                out.write(line)


def prepare(gpu_uuid: str) -> dict:
    """Validate the original pool and freeze data, token budgets and configs."""
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("a verified live GPU UUID is required")
    manifest_path = OUTPUT / "manifest.json"
    if manifest_path.exists():
        raise ValueError("sweep already prepared; verify rather than overwrite")
    base = yaml.safe_load((HERE / "config.yaml").read_text())
    for name, spec in base["inputs"].items():
        if file_hash(ROOT / spec["path"]) != spec["sha256"]:
            raise ValueError(f"original input drift: {name}")
    if file_hash(ROOT / CENSUS) != CENSUS_SHA:
        raise ValueError("token census drift")
    training_path = ROOT / base["inputs"]["training"]["path"]
    teacher_path = ROOT / base["inputs"]["teacher"]["path"]
    rows, targets = read_rows(training_path), read_rows(teacher_path)
    validate_targets(rows, targets)
    teachers = {identity(r): r for r in targets}
    census = {identity(r): r for r in read_rows(ROOT / CENSUS)}
    if len(rows) != 8688 or len(census) != 8688 or set(census) != set(teachers):
        raise ValueError("original population drift")
    groups = defaultdict(list)
    for r in rows:
        c = census[identity(r)]
        if (
            c["view"] != "original"
            or any(
                c[k] != r[k]
                for k in ["label", "trajectory_sha256", "student_prompt_sha256"]
            )
            or digest(r["student_prompt"]) != r["student_prompt_sha256"]
        ):
            raise ValueError("original prompt/token identity drift")
        groups[r["trajectory_sha256"]].append(identity(r))
    if sum(r["prompt_tokens"] for r in census.values()) != 83816369:
        raise ValueError("original total token drift")
    report = {
        "status": "prepared",
        "gpu_uuid": gpu_uuid,
        "seed": 0,
        "source_inputs": base["inputs"],
        "census_sha256": CENSUS_SHA,
        "threshold": 0.5,
        "cases": {},
        "files_sha256": {},
    }
    for name, delta in CASES.items():
        excluded = {
            identity(r)
            for r in rows
            if delta is not None
            and remove_disagreement(
                r["label"], teachers[identity(r)]["soft_target"], delta
            )
        }
        if len(excluded) != EXPECTED[name]:
            raise ValueError(f"frozen removal count drift: {name}")
        campaign = f"teacher-disagreement-{name}-sdpa04"
        data = ROOT / "data" / campaign
        if data.exists():
            raise ValueError(f"refuse to overwrite {data}")
        for source, filename in [
            (training_path, "training.jsonl"),
            (teacher_path, "teacher.jsonl"),
        ]:
            subset(source, data / filename, excluded)
        retained = read_rows(data / "training.jsonl")
        validate_targets(retained, read_rows(data / "teacher.jsonl"))
        lengths = [census[identity(r)]["prompt_tokens"] for r in retained]
        audit = {
            "rows": len(retained),
            "total": sum(lengths),
            "maximum": max(lengths),
            "configured_cap": 29696,
            "truncated": 0,
            "provenance": {"census": CENSUS, "sha256": CENSUS_SHA},
        }
        write_json(data / "token_audit.json", audit)
        removed = [r for r in rows if identity(r) in excluded]
        split_groups = [
            g
            for g in groups.values()
            if len(g) > 1 and 0 < sum(k in excluded for k in g) < len(g)
        ]
        selection = {
            "delta": delta,
            "removed": len(excluded),
            "retained": len(retained),
            "removed_by_source_label": dict(
                Counter(f"{r['raw_source']}/{r['label']}" for r in removed)
            ),
            "retained_by_source_label": dict(
                Counter(f"{r['raw_source']}/{r['label']}" for r in retained)
            ),
            "split_exact_content_groups": len(split_groups),
            "split_group_ids": split_groups,
            "ids": sorted(excluded),
            "token_audit": audit,
        }
        write_json(data / "filter.json", selection)
        config = copy.deepcopy(base)
        if name == "control":
            for field, spec in config["cached_control"]["inputs"].items():
                config["inputs"][field] = spec
                if file_hash(ROOT / spec["path"]) != spec["sha256"]:
                    raise ValueError(f"cached control drift: {field}")
        else:
            config.pop("cached_control")
        config.update(
            campaign_id=campaign,
            training_rows=len(retained),
            expected_steps=math.ceil(len(retained) / 32),
            expected_training_tokens=sum(lengths),
            gpu_uuid=gpu_uuid,
            merged_model="/tmp/gleipnir-merged/teacher-disagreement-sweep",
        )
        for field, filename in [
            ("training", "training.jsonl"),
            ("teacher", "teacher.jsonl"),
            ("token_audit", "token_audit.json"),
            ("filter", "filter.json"),
        ]:
            p = data / filename
            config["inputs"][field] = {
                "path": str(p.relative_to(ROOT)),
                "sha256": file_hash(p),
            }
        config_path = data / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        paths = [
            config_path,
            *(
                data / n
                for n in [
                    "training.jsonl",
                    "teacher.jsonl",
                    "token_audit.json",
                    "filter.json",
                ]
            ),
        ]
        report["files_sha256"].update(
            {str(p.relative_to(ROOT)): file_hash(p) for p in paths}
        )
        report["cases"][name] = {
            "config": str(config_path.relative_to(ROOT)),
            **selection,
        }
        print(
            f"filter_prepared {name} removed={len(excluded)} "
            f"retained={len(retained)} tokens={sum(lengths)} "
            f"split_groups={len(split_groups)}",
            flush=True,
        )
    report["source_sha256"] = {
        str(p.relative_to(ROOT)): file_hash(p)
        for p in sorted(HERE.iterdir())
        if p.is_file()
    }
    report["source_sha256"]["src/gleipnir/data/teacher_filter.py"] = file_hash(
        ROOT / "src/gleipnir/data/teacher_filter.py"
    )
    write_json(manifest_path, report)
    return report


def verify_preparation() -> dict:
    """Reject changed configs, membership, targets or sweep sources."""
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    for section in ["files_sha256", "source_sha256"]:
        for name, expected in manifest[section].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError(f"sweep preparation drift: {name}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    prepare(parser.parse_args().gpu_uuid)
