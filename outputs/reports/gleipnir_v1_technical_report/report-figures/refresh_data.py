"""Rebuild data.json, the small input snapshot of the held-out figures.

Every value is read from a result file of the stress tests (git-ignored), so
this step needs GLEIPNIR_DATA_ROOT pointing at the checkout that holds
`results/`. The plotting script reads only data.json.

    GLEIPNIR_DATA_ROOT=/path/to/gleipnir python refresh_data.py
"""

import json
import math
import os
from pathlib import Path

ROOT = Path(os.environ["GLEIPNIR_DATA_ROOT"])
RES = ROOT / "results"
HERE = Path(__file__).resolve().parent


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [max(0.0, (c - m) / d), min(1.0, (c + m) / d)]


def main() -> None:
    data: dict = {}

    # APPS, same text on both programs: pAUROC@20 with task-bootstrap intervals.
    bp = json.loads((RES / "figures/backdoor_pauroc.json").read_text())
    keys = {
        "4b": "4b-v03_student",
        "base4b": "4b-base_student",
        "27b": "27b-merged_teacher",
        "kimi": "kimi-k3",
        "sol": "gpt-6.1-sol",
    }
    conds = ["none", "irrelevant", "specific_accusation", "generic_accusation"]
    data["apps"] = {
        m: {
            c: {
                "value": bp[k][c]["pauroc20"],
                "ci95": bp[k][c]["ci95"],
                "tasks": bp[k][c]["tasks"],
            }
            for c in conds
        }
        for m, k in keys.items()
    }

    # Probe against output, AUROC with task-bootstrap intervals.
    data["probe"] = {}
    for m, arm in (("4b", "4b-v03_student"), ("base4b", "4b-base_student")):
        rep = json.loads(
            (RES / f"adversarial_crossed_text_probe/{arm}/report.json").read_text()
        )["conditions"]
        data["probe"][m] = {
            c: {
                s: {
                    "value": rep[c]["metrics"][s]["auroc"],
                    "ci95": rep[c]["metrics"][s]["auroc_ci95"],
                }
                for s in ("probe", "output")
            }
            for c in ("none", "specific_accusation", "generic_accusation")
        }

    # Lab liars: pAUROC@20 (seed bootstrap), liars flagged, and framed readers flagged.
    pa = json.loads((RES / "adversarial_lab_liars/pauroc20_posthoc.json").read_text())
    liars_flagged = {
        "4b": 0,
        "27b": 5,
        "qwen27b": 0,
        "kimi": 18,
    }  # adversarial_lab_liars README, line (a)
    framing_arms = {"4b": "v03-student", "27b": "27b-teacher", "kimi": "kimi-teacher"}
    fr = json.loads((RES / "adversarial_is_it_4b/report.json").read_text())["arms"]
    qb = json.loads((RES / "adversarial_is_it_4b/qwen-base/summary.json").read_text())
    lab = {}
    for m, k in (
        ("4b", "4b"),
        ("27b", "27b"),
        ("qwen27b", "qwen_base"),
        ("kimi", "kimi"),
    ):
        if m == "qwen27b":
            read_k, read_n = qb["accuse_w1"]
        else:
            f = fr[framing_arms[m]]["framing"]["accuse_w1"]
            read_n = f["readers"]
            read_k = round(f["flagged_1%"] * read_n)
        lab[m] = {
            "pauroc20": pa[k]["pauroc20"],
            "pauroc20_ci95": pa[k]["ci95"],
            "liars": [liars_flagged[m], 30],
            "liars_ci95": wilson(liars_flagged[m], 30),
            "readers": [read_k, read_n],
            "readers_ci95": wilson(read_k, read_n),
        }
    data["lab"] = lab

    out = HERE / "data.json"
    out.write_text(json.dumps(data, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
