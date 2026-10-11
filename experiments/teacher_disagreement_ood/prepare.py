"""Freeze OOD membership, cached comparisons and existing candidate identities."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from pathlib import Path

import yaml

from gleipnir.campaigns.monitoring.contract import Campaign
from gleipnir.data.monitoring import (
    digest,
    file_hash,
    read_rows,
    write_json,
    write_rows,
)
from gleipnir.evaluation.scoring import rendered

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NAME = "teacher-disagreement-strict-ood01"
DATA = ROOT / "data" / NAME
OUT = ROOT / "results" / NAME
PARENT = "results/teacher-disagreement-delta000-sdpa04"
SOURCE = "data/student_injection_awareness/regular/ood.jsonl"
SOURCE_SHA = "42556d4f6ef6f7715b3b27f0b4d6f11ae83ef9c8499009489799032c2abbbf73"
REFERENCES = {
    "gleipnir4b": (
        "results/tool_trajectory_distillation_ood/runs/4b/adapters/"
        "soft-n21837-mixed-qwen35-4b-seed0/predictions.jsonl"
    ),
    "unfiltered_regular": (
        "results/student_injection_awareness/4b/regular/ood/predictions.jsonl"
    ),
}
COUNTS = {
    "agentdojo": 1500,
    "cua_shade_arena": 955,
    "iac": 1443,
    "mle_sabotage": 148,
    "rogue_eval": 1499,
    "shade_arena": 850,
}


def bind_cached(values: list[dict], canonical: list[dict]) -> list[dict]:
    """Normalize cached scores only after an exact identity/label/prompt join."""
    if len(values) != len(canonical) or len({r["id"] for r in values}) != len(values):
        raise ValueError("cached reference coverage or uniqueness drift")
    bound = []
    for value, row in zip(values, canonical, strict=True):
        meta = row["metadata"]
        if not math.isfinite(value["score"]) or not 0 <= value["score"] <= 1:
            raise ValueError("invalid cached reference probability")
        if (
            value["id"] != row["id"]
            or value["label"] != meta["ground_truth"]
            or value["source"] != meta["source_dataset"]
            or value["source_prompt_sha256"] != digest(row["prompt"])
        ):
            raise ValueError("cached reference identity/label/source/prompt drift")
        bound.append({**value, **meta})
    return bound


def prepare() -> None:
    """Prepare on CPU without requiring local copies of GPU model weights."""
    from transformers import AutoTokenizer

    if DATA.exists():
        raise ValueError("OOD preparation already exists; refuse overwrite")
    if file_hash(ROOT / SOURCE) != SOURCE_SHA:
        raise ValueError("canonical prepared OOD input changed")
    rows = read_rows(ROOT / SOURCE)
    if (
        len(rows) != 6395
        or dict(Counter(r["metadata"]["source_dataset"] for r in rows)) != COUNTS
    ):
        raise ValueError("OOD source membership changed")
    references = {k: read_rows(ROOT / p) for k, p in REFERENCES.items()}
    for values in references.values():
        bind_cached(values, rows)
    parent_config = ROOT / PARENT / "config.yaml"
    config = copy.deepcopy(yaml.safe_load(parent_config.read_text()))
    train = read_rows(ROOT / config["inputs"]["training"]["path"])
    seen = {r["trajectory_sha256"] for r in train}
    if any(r["metadata"]["trajectory_sha256"] in seen for r in rows):
        raise ValueError("OOD/training trajectory overlap")
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "data/judge_injection_continuation/tokenizer", local_files_only=True
    )
    workload = []
    for start in range(0, len(rows), 64):
        batch = rows[start : start + 64]
        prompts = [rendered(tokenizer, row, "01") for row in batch]
        lengths = [
            len(x) for x in tokenizer(prompts, add_special_tokens=False)["input_ids"]
        ]
        for offset, (row, prompt, length) in enumerate(
            zip(batch, prompts, lengths, strict=True)
        ):
            for cached in references.values():
                old = cached[start + offset]
                if (
                    digest(prompt) != old["margin_prompt_sha256"]
                    or length != old["prompt_tokens"]
                    or not 0 < length < 32768
                ):
                    raise ValueError("frozen OOD rendering/token/context drift")
            workload.append(
                {
                    "id": row["id"],
                    "prompt": prompt,
                    "prompt_sha256": digest(prompt),
                    "prompt_tokens": length,
                    "metadata": row["metadata"],
                }
            )
    write_rows(DATA / "workload.jsonl", workload)
    config.update(
        campaign_id=NAME,
        evaluations=["ood"],
        baselines={},
        merged_model="/tmp/gleipnir-merged/teacher-disagreement-sweep",
        hypothesis=(
            "The fixed strict-filtered checkpoint's ID ranking gain transfers "
            "to the frozen OOD suite."
        ),
        intervention=(
            "Score only the completed strict delta=0 checkpoint; reuse cached "
            "Gleipnir 4B and regular references."
        ),
        selection=(
            "Fixed final strict checkpoint selected on ID before OOD; "
            "no OOD selection, retraining or promotion."
        ),
    )
    config["evaluation"]["populations"] = {"ood": 6395}
    for key, path in [
        ("ood", ROOT / SOURCE),
        ("ood_workload", DATA / "workload.jsonl"),
    ]:
        config["inputs"][key] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": file_hash(path),
        }
    config["parent_campaign"] = PARENT
    config["cached_references"] = REFERENCES
    config["extra_sources"] = [
        str(p.relative_to(ROOT)) for p in sorted(HERE.iterdir()) if p.is_file()
    ]
    (DATA / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    files = {
        SOURCE: SOURCE_SHA,
        str((DATA / "config.yaml").relative_to(ROOT)): file_hash(DATA / "config.yaml"),
        str((DATA / "workload.jsonl").relative_to(ROOT)): file_hash(
            DATA / "workload.jsonl"
        ),
    }
    files.update({p: file_hash(ROOT / p) for p in REFERENCES.values()})
    files.update({spec["path"]: spec["sha256"] for spec in config["inputs"].values()})
    for name in [
        "config.yaml",
        "merged_artifact.json",
        "merged_parity.json",
        "canary_workload.jsonl",
    ]:
        p = ROOT / PARENT / name
        if p.exists():
            files[str(p.relative_to(ROOT))] = file_hash(p)
    # Model weights remain remote: bind them to the successful immutable stage receipt.
    runner = json.loads((ROOT / PARENT / "runner.json").read_text())
    for stage in ["train", "master-reference", "merged-reference"]:
        if runner["stages"][stage]["status"] != "complete":
            raise ValueError("candidate did not complete its original gates")
        for path, expected in runner["stages"][stage]["files_sha256"].items():
            if path.startswith("/workspace/gleipnir/"):
                files[path.removeprefix("/workspace/gleipnir/")] = expected
    artifact = json.loads((ROOT / PARENT / "merged_artifact.json").read_text())
    files.update(
        {
            config["base_model"] + "/" + name: sha
            for name, sha in artifact["source_files_sha256"].items()
        }
    )
    sources = Campaign(ROOT, DATA / "config.yaml", config).sources()
    write_json(
        DATA / "manifest.json",
        {
            "files_sha256": files,
            "source_sha256": sources,
            "rows": len(rows),
            "source_rows": COUNTS,
            "tokens": sum(r["prompt_tokens"] for r in workload),
            "maximum_tokens": max(r["prompt_tokens"] for r in workload),
            "prior_ood_use": (
                "Existing repeatedly evaluated six-source suite; no new holdout claim"
            ),
            "selection": "Strict delta=0 fixed on ID before OOD; no promotion",
        },
    )
    print(
        f"ood_prepared rows={len(rows)} "
        f"tokens={sum(r['prompt_tokens'] for r in workload)} "
        f"max={max(r['prompt_tokens'] for r in workload)}",
        flush=True,
    )


if __name__ == "__main__":
    prepare()
