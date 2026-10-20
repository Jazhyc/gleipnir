"""Run the separately frozen stronger-writer A2 comparison."""

import argparse
import json
from pathlib import Path

from experiments.adversarial_a2_cover_comments import run as shared
from gleipnir.adversarial import eval_sets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "score", "report"])
    args = parser.parse_args()
    shared.HERE = Path(__file__).resolve().parent
    shared.CONFIG = json.loads((shared.HERE / "config.json").read_text())
    shared.OUT = eval_sets.ROOT / shared.CONFIG["output"]
    getattr(shared, args.command)()
