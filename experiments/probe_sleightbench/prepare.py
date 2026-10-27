"""Materialize NTP rows for the SleightBench probe pass (CPU, no model).

Writes two splits under ``data/probe_generalization/sleightbench/`` so the
probe study's ``extract.py`` reads them unchanged:

- ``stripped``: the 86 graded SleightBench transcripts (43 attacks, 43 benign
  twins) in the frozen ``stripped`` view of ``experiments/sleightbench_eval``.
  The NTP input is the trajectory body that the released monitors saw inside
  ``<agent_trajectory>``, with no instruction or chat template.
- ``parity_id``: probe-study ID rows (the longest ones plus a seeded sample)
  whose stored probe logits check the re-built extraction environment.

The frozen manifest (``manifest.json`` beside this file) records hashes,
labels and lengths only, never benchmark text.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from experiments.sleightbench_eval import benchmark
from gleipnir.data.monitoring import digest, file_hash, write_json, write_rows

EXPERIMENT = Path(__file__).parent
ROOT = EXPERIMENT.parents[1]
SLEIGHT = json.loads((ROOT / "experiments/sleightbench_eval/config.json").read_text())
SLEIGHT_DATA = ROOT / SLEIGHT["upstream"]["data_root"]
PROBE_DATA = ROOT / "data/probe_generalization"
OUT = PROBE_DATA / "sleightbench"
VIEW = "stripped"
PARITY_LONGEST = 8
PARITY_SAMPLED = 8
SEED = 0


def stripped_rows() -> list[dict]:
    hashes = json.loads((SLEIGHT_DATA / "decrypted_sha256.json").read_text())
    for name, expected in hashes.items():
        if file_hash(SLEIGHT_DATA / "attacks" / name) != expected:
            raise ValueError(f"decrypted SleightBench file changed: {name}")
    rows = []
    for t in benchmark.transcripts(SLEIGHT_DATA, set(SLEIGHT["excluded_attacks"])):
        trajectory = benchmark.body(
            benchmark.VIEWS[VIEW](benchmark.read_messages(t.path))
        )
        rows.append(
            {
                "id": f"{VIEW}/{t.id}",
                "split": VIEW,
                "source": "sleightbench",
                "label": t.label,
                "attack": t.attack,
                "category": t.category,
                "trajectory": trajectory,
                "trajectory_sha256": digest(trajectory),
            }
        )
    return rows


def parity_rows() -> list[dict]:
    rows = [json.loads(line) for line in (PROBE_DATA / "full/id.jsonl").open()]
    by_length = sorted(rows, key=lambda r: (-len(r["trajectory"]), r["id"]))
    chosen = by_length[:PARITY_LONGEST]
    rest = sorted(by_length[PARITY_LONGEST:], key=lambda r: r["id"])
    chosen += random.Random(SEED).sample(rest, PARITY_SAMPLED)
    return [
        {
            key: r[key]
            for key in ("id", "source", "label", "trajectory", "trajectory_sha256")
        }
        | {"split": "parity_id"}
        for r in chosen
    ]


def main() -> None:
    splits = {VIEW: stripped_rows(), "parity_id": parity_rows()}
    labels = [r["label"] for r in splits[VIEW]]
    if len(labels) != 86 or sum(labels) != 43:
        raise ValueError(f"expected 43 attack and 43 benign rows, got {labels}")
    manifest: dict = {
        "view": VIEW,
        "upstream": SLEIGHT["upstream"],
        "sleightbench_manifest_sha256": file_hash(
            ROOT / "experiments/sleightbench_eval/manifest.json"
        ),
        "probe_id_sha256": file_hash(PROBE_DATA / "full/id.jsonl"),
        "parity_rule": {
            "longest": PARITY_LONGEST,
            "sampled": PARITY_SAMPLED,
            "seed": SEED,
        },
        "splits": {},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        path = OUT / f"{split}.jsonl"
        write_rows(path, rows)
        manifest["splits"][split] = {
            "sha256": file_hash(path),
            "rows": [
                {
                    "id": r["id"],
                    "label": r["label"],
                    "chars": len(r["trajectory"]),
                    "trajectory_sha256": r["trajectory_sha256"],
                }
                for r in rows
            ],
        }
    path = EXPERIMENT / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
            raise ValueError("prepared rows differ from the frozen manifest")
        print("manifest unchanged")
    else:
        write_json(path, manifest)
        print(f"froze {sum(len(s) for s in splits.values())} rows")


if __name__ == "__main__":
    main()
