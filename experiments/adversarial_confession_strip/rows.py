"""Build the stripped rows from monitor-aware's scoring rows (CPU, no network).

    PYTHONPATH=src:. ../../.venv-data/bin/python \
        experiments/adversarial_confession_strip/rows.py

For each of monitor-aware's 413 rows: take the run's messages up to the row's
cut, check that rendering them reproduces monitor-aware's prompt byte for
byte, strip them (`strip.py`), re-render under the same prompt head, and scan
the stripped cut for admission words. Writes `rows_student.jsonl`,
`rows_teacher.jsonl`, `canaries.jsonl` (monitor-aware's, unchanged) and
`manifest.json` to `data/adversarial_confession_strip/`. Row ids, cuts and
too_long flags are monitor-aware's.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.adversarial_confession_strip.strip import strip_messages, summary
from experiments.adversarial_natural_cheats.trajectories import render

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = json.loads((HERE / "config.json").read_text())
SOURCE = CONFIG["source"]
SOURCE_DATA = ROOT / SOURCE["data"]
DATA = ROOT / CONFIG["data"]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return file_sha256(path)


def checked(path: Path, digest: str) -> Path:
    if file_sha256(path) != digest:
        raise ValueError(f"{path} does not match its frozen SHA256")
    return path


def heads() -> dict[str, str]:
    out = {}
    for name, p in SOURCE["prompts"].items():
        text = (ROOT / p).read_text()
        if sha(text) != SOURCE["prompt_sha256"][name]:
            raise ValueError(f"{name} prompt does not match its frozen SHA256")
        out[name] = text
    return out


def admits(scan: dict[str, list[str]]) -> bool:
    return bool(scan["arguments"] or scan["outputs"] or scan["messages"])


def build_one(
    run: dict[str, Any], meta: dict[str, Any], known: set[str], patterns: list[str]
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """(original cut, stripped cut, the row's strip record)."""
    from experiments.adversarial_quiet_cheats.checks import admission_scan

    kept = run["messages"][: meta["cut_message_index"] + 1]
    if len(kept) != meta["messages_in_cut"]:
        raise ValueError(f"{run['id']}: cut length differs from monitor-aware")
    stats: Counter[str] = Counter()
    stripped = strip_messages(kept, stats)
    before, after = (
        admission_scan(kept, known, patterns),
        admission_scan(stripped, known, patterns),
    )
    record = {
        **summary(stats),
        "trajectory_chars_before": len(render(kept)),
        "trajectory_chars_after": len(render(stripped)),
        "admission_before": before,
        "admission_after": after,
        "admits_before": admits(before),
        "admits_after": admits(after),
    }
    return kept, stripped, record


def main() -> None:
    from experiments.adversarial_quiet_cheats.checks import task_lines
    from gleipnir.adversarial.contract import join_prompt

    checked(SOURCE_DATA / "runs.jsonl", SOURCE["runs_sha256"])
    runs = {r["id"]: r for r in read_jsonl(SOURCE_DATA / "runs.jsonl")}
    source = {
        p: read_jsonl(
            checked(SOURCE_DATA / f"rows_{p}.jsonl", SOURCE["rows_sha256"][f"rows_{p}"])
        )
        for p in ("student", "teacher")
    }
    if [r["id"] for r in source["student"]] != [r["id"] for r in source["teacher"]]:
        raise ValueError("student and teacher rows differ in order")
    texts = heads()
    patterns = CONFIG["admission_patterns"]
    out: dict[str, list[dict]] = {"student": [], "teacher": []}
    totals: Counter[str] = Counter()
    by_label: dict[str, Counter] = {"cheat": Counter(), "non_cheat": Counter()}
    for i, s_row in enumerate(source["student"]):
        meta = s_row["metadata"]
        run = runs[meta["parent_id"]]
        known = task_lines(run["prompt"], run["test"], run["entry_point"])
        kept, stripped, record = build_one(run, meta, known, patterns)
        for p in ("student", "teacher"):
            original = source[p][i]
            if join_prompt(texts[p], render(kept)) != original["prompt"]:
                raise ValueError(f"{original['id']}: re-render differs ({p})")
            prompt = join_prompt(texts[p], render(stripped))
            out[p].append(
                {
                    "id": original["id"],
                    "prompt": prompt,
                    "metadata": {
                        **original["metadata"],
                        "source_dataset": "adversarial_confession_strip",
                        "original_prompt_sha256": sha(original["prompt"]),
                        "strip": record,
                    },
                }
            )
        group = by_label["cheat" if meta["cheat"] else "non_cheat"]
        group["rows"] += 1
        group["rows_changed"] += render(kept) != render(stripped)
        group["admits_before"] += record["admits_before"]
        group["admits_after"] += record["admits_after"]
        group["admits_after_in_arguments"] += bool(
            record["admission_after"]["arguments"]
        )
        group["admits_after_in_outputs"] += bool(record["admission_after"]["outputs"])
        group["admits_after_outputs_only"] += bool(
            record["admission_after"]["outputs"]
            and not record["admission_after"]["arguments"]
        )
        for k, v in record.items():
            if isinstance(v, int) and not isinstance(v, bool):
                totals[k] += v
                group[k] += v
        for k, v in record["detail"].items():
            totals[f"detail:{k}"] += v
    DATA.mkdir(parents=True, exist_ok=True)
    digests = {f"rows_{p}": write_jsonl(DATA / f"rows_{p}.jsonl", out[p]) for p in out}
    canaries = DATA / "canaries.jsonl"
    shutil.copyfile(
        checked(ROOT / CONFIG["canary"]["source"], CONFIG["canary"]["sha256"]), canaries
    )
    digests["canaries"] = file_sha256(canaries)
    hit_words: dict[str, Counter] = {"arguments": Counter(), "outputs": Counter()}
    for row in out["student"]:
        if row["metadata"]["cheat"]:
            for place in hit_words:
                hit_words[place].update(
                    row["metadata"]["strip"]["admission_after"][place]
                )
    manifest = {
        "rows": len(out["student"]),
        "per_label": {k: dict(v) for k, v in by_label.items()},
        "totals": dict(sorted(totals.items())),
        "cheat_admission_words_after": {k: dict(v) for k, v in hit_words.items()},
        "scorable": {
            "4b": sum(not r["metadata"]["too_long"]["4b"] for r in out["student"]),
            "27b": sum(not r["metadata"]["too_long"]["27b"] for r in out["teacher"]),
        },
        "sha256": digests,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
