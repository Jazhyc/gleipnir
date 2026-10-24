"""Materialize probe-study rows for both backbones (CPU, stdlib only).

Each output row carries the teacher-prompt user message (the *prompted*
backbone; ``trajectory_start`` is the character offset where the trajectory
begins), the bare trajectory (the *NTP* backbone), the source label and, for
training rows, Kimi K3's soft target. ``--population pilot`` draws the frozen
pilot sample; ``--population full`` keeps every row.

Splits: ``train`` (the 7,836 strict-filtered rows), ``id`` (the canonical
CoT-removed ID rows re-rendered with the teacher template) and ``ood`` (the
frozen OOD suite). The pilot's ``parity`` split is 128 OOD rows with the
archived stock-vLLM base-4B margins, used only for the extraction parity check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from experiments.tool_trajectory_monitoring.prompting import load_prompt_set
from gleipnir.data.monitoring import (
    digest,
    file_hash,
    read_rows,
    rerender_evaluation,
    trajectory_from_prompt,
    write_json,
    write_rows,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/probe_generalization"
INPUTS = {
    "train_prompts": (
        "data/tool_trajectory_monitoring/teacher_training/prompts.jsonl",
        "c06833bc0c5e8893ea45e92ec642a71e751749ee1cd0796eb88f7ab38402763f",
    ),
    "strict_rows": ("data/qwen38-27b-s0-01/strict_filter/training.jsonl", None),
    "strict_targets": ("data/qwen38-27b-s0-01/strict_filter/teacher.jsonl", None),
    "id_prompts": (
        "data/id_cot_only_evaluation/prompts.jsonl",
        "7a4ec604740b4b3b7bd70aafd4473b1b6bfcbf92706bcd23b90ed8df2c4bc1b4",
    ),
    "ood_prompts": (
        "data/tool_trajectory_monitoring/teacher_ood_benchmark/prompts.jsonl",
        "fa4b0d917e51eb91a52efa93cddf0eb154b88993269c97cfdb161c42680136a9",
    ),
    "ood_base_predictions": (
        "results/tool_trajectory_monitoring/qwen35_4b_teacher_ood/predictions.jsonl",
        None,
    ),
}
PILOT = {"train": 1000, "id": 1000, "parity": 128}
SEED = 0


def load_inputs() -> tuple[dict[str, list[dict]], dict[str, dict]]:
    rows, provenance = {}, {}
    for name, (path, expected) in INPUTS.items():
        full = ROOT / path
        sha = file_hash(full)
        if expected is not None and sha != expected:
            raise ValueError(f"frozen input changed: {path}")
        rows[name] = read_rows(full)
        provenance[name] = {"path": path, "sha256": sha, "rows": len(rows[name])}
    return rows, provenance


def row(
    split: str, source_row: dict, trajectory: str, prompt: str, label: int, **extra
) -> dict:
    teacher = load_prompt_set().teacher
    if prompt != teacher.render(trajectory):
        raise ValueError(f"teacher envelope drift in {source_row['id']}")
    return {
        "id": str(source_row["id"]),
        "split": split,
        "source": str(source_row["metadata"]["source_dataset"]),
        "label": int(label),
        "prompt": prompt,
        "trajectory_start": len(teacher.cache_prefix),
        "trajectory": trajectory,
        "trajectory_sha256": digest(trajectory),
        **extra,
    }


def build_train(inputs: dict[str, list[dict]]) -> list[dict]:
    teacher = load_prompt_set().teacher
    prompts = {r["id"]: r for r in inputs["train_prompts"]}
    targets = {r["index"]: r for r in inputs["strict_targets"]}
    out = []
    for strict in inputs["strict_rows"]:
        source = prompts[strict["index"]]
        target = targets[strict["index"]]
        meta = source["metadata"]
        rendered = digest(source["prompt"])
        if not (
            rendered
            == meta["rendered_prompt_sha256"]
            == strict["teacher_rendered_prompt_sha256"]
            == target["rendered_prompt_sha256"]
        ):
            raise ValueError(f"teacher prompt identity drift in {strict['index']}")
        if not int(meta["ground_truth"]) == strict["label"] == target["label"]:
            raise ValueError(f"label drift in {strict['index']}")
        trajectory = trajectory_from_prompt(
            source["prompt"], meta["trajectory_sha256"], source_template=teacher
        )
        out.append(
            row(
                "train",
                source,
                trajectory,
                source["prompt"],
                strict["label"],
                soft_target=float(target["soft_target"]),
                teacher_model=target["teacher_model"],
            )
        )
    return out


def build_eval(split: str, source_rows: list[dict], rerender: bool) -> list[dict]:
    teacher = load_prompt_set().teacher
    rendered = rerender_evaluation(source_rows, teacher) if rerender else source_rows
    out = []
    for source in rendered:
        meta = source["metadata"]
        if digest(source["prompt"]) != meta["rendered_prompt_sha256"]:
            raise ValueError(f"evaluation prompt drift in {source['id']}")
        trajectory = trajectory_from_prompt(
            source["prompt"], meta["trajectory_sha256"], source_template=teacher
        )
        out.append(
            row(split, source, trajectory, source["prompt"], meta["ground_truth"])
        )
    return out


def stratified(rows: list[dict], count: int, keys: tuple[str, ...]) -> list[dict]:
    """Proportional allocation by stratum (largest remainder), seeded."""
    strata = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        strata[tuple(r[k] for k in keys)].append(r)
    quotas = {k: count * len(v) / len(rows) for k, v in strata.items()}
    taken = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: quotas[k] - taken[k], reverse=True)[
        : count - sum(taken.values())
    ]:
        taken[k] += 1
    rng = random.Random(SEED)
    sample = []
    for k in sorted(strata):
        sample.extend(rng.sample(strata[k], taken[k]))
    return sorted(sample, key=lambda r: r["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    inputs, provenance = load_inputs()
    splits = {
        "train": build_train(inputs),
        "id": build_eval("id", inputs["id_prompts"], rerender=True),
        "ood": build_eval("ood", inputs["ood_prompts"], rerender=False),
    }
    if {r["trajectory_sha256"] for r in splits["train"]} & {
        r["trajectory_sha256"] for r in splits["id"]
    }:
        raise ValueError("training and ID trajectories overlap")
    if args.population == "pilot":
        margins = {
            str(p["id"]): p["logit_margin_1_minus_0"]
            for p in inputs["ood_base_predictions"]
        }
        parity = stratified(splits.pop("ood"), PILOT["parity"], ("source",))
        for r in parity:
            r["split"] = "parity"
            r["reference_margin"] = float(margins[r["id"]])
        splits = {
            "train": stratified(splits["train"], PILOT["train"], ("source", "label")),
            "id": stratified(splits["id"], PILOT["id"], ("source", "label")),
            "parity": parity,
        }
    out = OUT / args.population
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"population": args.population, "seed": SEED, "inputs": provenance}
    manifest["teacher_template_sha256"] = load_prompt_set().teacher.template_sha256
    manifest["splits"] = {}
    for split, rows in splits.items():
        path = out / f"{split}.jsonl"
        write_rows(path, rows)
        counts = defaultdict(int)
        for r in rows:
            counts[f"{r['source']}/{r['label']}"] += 1
        manifest["splits"][split] = {
            "rows": len(rows),
            "sha256": file_hash(path),
            "ids_sha256": hashlib.sha256(
                "\n".join(r["id"] for r in rows).encode()
            ).hexdigest(),
            "trajectory_chars": sum(len(r["trajectory"]) for r in rows),
            "by_source_label": dict(sorted(counts.items())),
        }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
