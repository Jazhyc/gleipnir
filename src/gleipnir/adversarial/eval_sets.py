"""Frozen evaluation sets, located under `data/` and checked by SHA256.

ID and OOD come from `scripts/build_eval_datasets.py`. APPS comes from
`Jazhyc/gleipnir-apps-injection-evals` at `APPS_REVISION`, downloaded with
`allow_patterns=["data/*"]` into the repository root.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[3]
APPS_REPO = "Jazhyc/gleipnir-apps-injection-evals"
APPS_REVISION = "f094c15454e2f07782d58760c519835335693ad3"


class EvalSet(NamedTuple):
    path: str
    sha256: str
    rows: int


SETS = {
    "id": EvalSet(
        "data/id_cot_only_evaluation/prompts.jsonl",
        "7a4ec604740b4b3b7bd70aafd4473b1b6bfcbf92706bcd23b90ed8df2c4bc1b4",
        3012,
    ),
    "ood": EvalSet(
        "data/tool_trajectory_monitoring/distillation_ood/prompts.jsonl",
        "ee2baa18a5d36cda8f9d66df642b6f3c106f92ae8ce2813acca36fa9f0e4acd2",
        6395,
    ),
    "apps-neutral": EvalSet(
        "data/apps/prompts/eval_neutral.jsonl",
        "339327a476f6834d8998500e2726263dfe3ce55bcc0a12648928abe9e99abfc0",
        5184,
    ),
    "apps-aware": EvalSet(
        "data/apps/prompts/eval_injection_aware.jsonl",
        "b376a701853d4189a300aaf3d310e949256f38b2619b28017d362c446c284eec",
        5184,
    ),
    "apps-controls-neutral": EvalSet(
        "data/apps/honest_controls/eval_neutral.jsonl",
        "71b337c070fc60faba0d04c54c5f0813498c45d60ab6b9c77b9b67639c6a70e1",
        3930,
    ),
    "apps-controls-aware": EvalSet(
        "data/apps/honest_controls/eval_injection_aware.jsonl",
        "6c90d47d6c0fac4cdc2b97803e004a42d49cd7488c4c8f71bbb5f894ac802369",
        3930,
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path(name: str, root: Path = ROOT) -> Path:
    return root / SETS[name].path


def load(name: str, root: Path = ROOT) -> list[dict[str, Any]]:
    """Read a set; refuse a file whose hash or row count drifted."""
    spec = SETS[name]
    file = root / spec.path
    if file_sha256(file) != spec.sha256:
        raise ValueError(f"{name}: {file} does not match its frozen SHA256")
    rows = [json.loads(line) for line in file.open()]
    if len(rows) != spec.rows:
        raise ValueError(f"{name}: expected {spec.rows} rows, found {len(rows)}")
    return rows


def label(row: dict[str, Any]) -> int:
    return int(row["metadata"]["ground_truth"])


def group(row: dict[str, Any]) -> str:
    """OOD/ID source, or APPS condition."""
    metadata = row["metadata"]
    return str(metadata.get("condition") or metadata["source_dataset"])
