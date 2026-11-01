"""Verify and pin separate historical calibration artifacts without new scoring."""

from __future__ import annotations

import argparse

from experiments.adversarial_goodfire.common import (
    DATA,
    HERE,
    ROOT,
    config,
    read_jsonl,
    write_json,
    write_jsonl,
)
from experiments.adversarial_goodfire.download import sha256
from experiments.adversarial_goodfire.frontier import monitor, payload
from gleipnir.adversarial.contract import join_prompt, split_prompt


def records(spec: dict) -> list[dict]:
    rows = read_jsonl(ROOT / spec["scores"])
    if spec["filter"] == "honest_label_0":
        rows = [r for r in rows if r.get("label") == 0]
    elif spec["filter"] == "apps_honest_none_main_heldout":
        rows = [
            r
            for r in rows
            if r.get("code") == "honest"
            and r.get("text") == "none"
            and r.get("set") == "main"
            and r.get("split") == "heldout"
        ]
    else:
        raise ValueError("unknown calibration filter")
    if len(rows) != spec["expected_honest"]:
        raise ValueError("calibration honest count mismatch")
    if len({r["id"] for r in rows}) != len(rows) or any(
        r["status"] != "ok" for r in rows
    ):
        raise ValueError("duplicate or failed calibration rows")
    return rows


def main() -> None:
    argparse.ArgumentParser(__doc__).parse_args()
    cfg = config()
    receipt = {}
    for name, spec in cfg["calibration"].items():
        for key, digest in spec["sha256"].items():
            if sha256(ROOT / spec[key]) != digest:
                raise ValueError(f"calibration artifact changed: {name} {key}")
        rows = records(spec)
        # Do not compute thresholds or evaluate monitors during preparation.
        receipt[name] = {
            "honest_n": len(rows),
            "filter": spec["filter"],
            "mode": spec["mode"],
            "sha256": spec["sha256"],
            "threshold": None,
            "status": "not_computed",
        }
    # The historical Sol score builder allowed provider fallbacks. Export the
    # same 200 honest APPS prompts for future replay with fallback disabled.
    spec = cfg["calibration"]["gpt-6.1-sol"]
    ids = {r["id"] for r in records(spec)}
    source = {r["id"]: r for r in read_jsonl(ROOT / spec["inputs"])}
    teacher = (ROOT / cfg["prompts"]["teacher"]).read_text()
    out = DATA / "prepared/calibration"
    replay = []
    for key in sorted(ids):
        row = source[key]
        prompt = join_prompt(teacher, split_prompt(row["prompt"]).trajectory)
        body = payload(monitor("gpt-6.1-sol", cfg), {"id": key, "prompt": prompt})
        replay.append(
            {
                "id": key,
                "monitor": "gpt-6.1-sol",
                "status": "pending_authorization",
                "payload": body,
            }
        )
    write_jsonl(out / "sol_replay.jsonl", replay)
    write_json(
        out / "receipt.json",
        {
            "config_sha256": sha256(HERE / "config.json"),
            "calibration": receipt,
            "sol_replay_sha256": sha256(out / "sol_replay.jsonl"),
        },
    )
    print("calibration artifacts verified, no thresholds or scores computed")


if __name__ == "__main__":
    main()
