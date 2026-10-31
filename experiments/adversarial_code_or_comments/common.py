"""Shared paths, frozen inputs, JSONL helpers and run guards for this experiment."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from gleipnir.adversarial import apps, eval_sets

HERE = Path(__file__).resolve().parent
ROOT = eval_sets.ROOT
CONFIG_PATH = HERE / "config.json"
PROMPTS = HERE / "prompts"
CODES = ("honest", "backdoored")


def config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def order_key(seed: int, *parts: str) -> str:
    return sha(":".join([str(seed), *map(str, parts)]))


def output_dir(path: str | None, cfg: dict[str, Any]) -> Path:
    """`--out` if given, else the experiment's ignored results folder."""
    out = Path(path) if path else ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def prompt(name: str, cfg: dict[str, Any] | None = None) -> str:
    """A prompt file, checked against its frozen hash once one is recorded."""
    text = (PROMPTS / name).read_text()
    frozen = (cfg or config()).get("prompt_sha256", {}).get(name)
    if frozen is not None and frozen != sha(text):
        raise ValueError(f"prompt {name} does not match its frozen SHA256")
    return text


def fill(template: str, **values: Any) -> str:
    """Replace `{name}` placeholders only, so braces inside values are safe."""
    for name, value in values.items():
        template = template.replace("{" + name + "}", str(value))
    return template


def instruction(name: str, cfg: dict[str, Any]) -> str:
    """The student or teacher monitor instruction, checked by SHA256."""
    data = cfg["data"]
    text = (ROOT / data[f"{name}_instruction"]).read_text()
    if sha(text) != data[f"{name}_instruction_sha256"]:
        raise ValueError(f"{name} instruction does not match its frozen SHA256")
    return text


def apps_pairs(rows: list[dict[str, Any]], cfg: dict[str, Any]):
    """Task id -> {"honest": row, "backdoored": row} for tasks with both.

    Returns the pairs and counts of tasks seen per code, so the report can show
    what was dropped.
    """
    conditions = {
        cfg["data"]["honest_condition"]: "honest",
        cfg["data"]["backdoor_condition"]: "backdoored",
    }
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        code = conditions.get(eval_sets.group(row))
        if code is None:
            continue
        task = str(row["metadata"]["task_id"])
        if code in by_task.setdefault(task, {}):
            raise ValueError(f"task {task} has two {code} rows")
        by_task[task][code] = row
    counts = {code: sum(code in d for d in by_task.values()) for code in CODES}
    pairs = {}
    for task, found in by_task.items():
        if len(found) != 2:
            continue
        try:
            for code in CODES:
                compile(apps.parse(found[code]["prompt"]).code, task, "exec")
        except (ValueError, SyntaxError):
            continue
        pairs[task] = found
    counts["both"] = len(pairs)
    return pairs, counts


def load_pairs(cfg: dict[str, Any]):
    data = cfg["data"]
    spec = eval_sets.SETS[data["set"]]
    if (spec.sha256, spec.rows, spec.path) != (
        data["sha256"],
        data["rows"],
        data["path"],
    ):
        raise ValueError("config data block disagrees with eval_sets")
    return apps_pairs(eval_sets.load(data["set"]), cfg)


def load_key(name: str, cfg: dict[str, Any]) -> str:
    """One key from the environment or the wrapper .env; never printed."""
    if os.environ.get(name):
        return os.environ[name]
    env = (ROOT / cfg["env_file"]).resolve()
    for line in env.read_text().splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{name} missing from the environment and {env.name}")


def git_clean(*paths: str) -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return not out.stdout.strip()


def require_frozen(cfg: dict[str, Any]) -> None:
    """Paid or GPU stages run only from a committed, frozen protocol."""
    if cfg.get("protocol_status") != "frozen":
        raise SystemExit("protocol_status is not 'frozen'; commit the protocol first")
    folders = (
        "experiments/adversarial_code_or_comments",
        "src/gleipnir/adversarial",
    )
    if not git_clean(*folders):
        raise SystemExit("uncommitted changes in the protocol; commit them first")


def require_predictions(cfg: dict[str, Any]) -> None:
    """Held-out scoring needs committed predictions."""
    path = cfg["analysis"]["predictions_file"]
    if not (ROOT / path).exists():
        raise SystemExit(f"{path} is missing; freeze predictions before held-out")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path], cwd=ROOT, capture_output=True
    )
    if tracked.returncode or not git_clean(path):
        raise SystemExit(f"{path} is not committed")
