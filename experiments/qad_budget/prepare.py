"""Nested proportionally stratified views of the existing same-prompt BF16 cache."""

import bisect
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from gleipnir.data.monitoring import file_hash, read_rows, write_json, write_rows


def select(workload, teachers, count: int, seed: int):
    """Return a deterministic quota-balanced stream and its original strata."""
    indexed = {(row["dataset"], row["index"]): row for row in teachers}
    identities = [(row["dataset"], row["id"]) for row in workload]
    if len(indexed) != len(teachers) or len(set(identities)) != len(workload):
        raise ValueError("duplicate source identity")
    if set(indexed) != set(identities) or not 0 < count <= len(workload):
        raise ValueError("teacher membership or subset size mismatch")
    lengths = sorted(row["prompt_tokens"] for row in workload)
    cuts = [lengths[int((len(lengths) - 1) * q)] for q in (0.25, 0.5, 0.75)]
    buckets, strata = defaultdict(list), []
    for i, (row, identity) in enumerate(zip(workload, identities, strict=True)):
        target = indexed[identity]
        p = float(target["soft_target"])
        if (
            target["rendered_prompt_sha256"] != row["prompt_sha256"]
            or target["label"] != row["label"]
            or not math.isfinite(p)
            or not 0 <= p <= 1
        ):
            raise ValueError("teacher prompt/label/probability mismatch")
        key = (
            row["source_dataset"],
            row["label"],
            bisect.bisect_right((0.1, 0.4, 0.6, 0.9), p),
            bisect.bisect_right(cuts, row["prompt_tokens"]),
        )
        buckets[key].append(i)
        strata.append(key)
    rng = random.Random(seed)
    population = {key: len(values) for key, values in buckets.items()}
    for key in sorted(buckets):
        rng.shuffle(buckets[key])
    used, order = Counter(), []
    for step in range(1, count + 1):
        choices = [key for key in sorted(buckets) if used[key] < population[key]]
        key = max(
            choices, key=lambda key: step * population[key] / len(workload) - used[key]
        )
        order.append(buckets[key][used[key]])
        used[key] += 1
    return (
        order,
        strata,
        {
            "length_quartiles": cuts,
            "population": {str(k): v for k, v in population.items()},
        },
    )


def prepare(root: Path, config):
    parent = root / "data/quantization-recovery01"
    if file_hash(parent / "teacher.jsonl") != config["teacher_sha256"]:
        raise ValueError("full BF16 teacher drift")
    if file_hash(parent / "workload.jsonl") != config["inputs"]["workload"]["sha256"]:
        raise ValueError("full workload drift")
    workload, teachers = (
        read_rows(parent / "workload.jsonl"),
        read_rows(parent / "teacher.jsonl"),
    )
    order, strata, audit = select(workload, teachers, 2048, config["seed"])
    indexed = {(r["dataset"], r["index"]): r for r in teachers}
    output = root / "results" / config["experiment"]
    if (output / "selection.json").exists():
        raise ValueError("preserve existing frozen subset selection")
    paths = []
    for updates in (32, 64):
        name = f"qad-budget{updates}-01"
        chosen = order[: updates * 32]
        rows = [workload[i] for i in chosen]
        targets = [indexed[(r["dataset"], r["id"])] for r in rows]
        data = root / "data" / name
        write_rows(data / "workload.jsonl", rows)
        write_rows(data / "teacher.jsonl", targets)
        derived_sha = file_hash(data / "teacher.jsonl")
        write_json(
            data / "teacher_complete.json",
            {
                "teacher_sha256": derived_sha,
                "parent_teacher_sha256": config["teacher_sha256"],
                "rows": len(rows),
                "scope": "subset_of_completed_cache_no_new_requests",
            },
        )
        arm = {
            **config,
            "experiment": name,
            "training_rows": len(rows),
            "training_tokens": sum(r["prompt_tokens"] for r in rows),
            "updates": updates,
            "teacher_experiment": name,
            "teacher_sha256": derived_sha,
            "inputs": {
                **config["inputs"],
                "workload": {
                    "path": str((data / "workload.jsonl").relative_to(root)),
                    "sha256": file_hash(data / "workload.jsonl"),
                },
            },
            "merge_predecessor": config["merge_predecessor"]
            if updates == 32
            else "results/qad-budget32-01/merged_artifact.json",
        }
        path = output / f"arm{updates}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(arm, sort_keys=False))
        paths.append(path)
        write_json(
            output / f"subset{updates}.json",
            {
                "ids": [r["id"] for r in rows],
                "rows": len(rows),
                "tokens": arm["training_tokens"],
                "strata": dict(Counter(str(strata[i]) for i in chosen)),
                "teacher_sha256": derived_sha,
                "workload_sha256": arm["inputs"]["workload"]["sha256"],
                "config_sha256": file_hash(path),
            },
        )
    write_json(
        output / "selection.json",
        {
            **audit,
            "seed": config["seed"],
            "nested": True,
            "parent_teacher_sha256": config["teacher_sha256"],
            "order": order,
        },
    )
    return paths
