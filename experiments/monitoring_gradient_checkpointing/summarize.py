#!/usr/bin/env python3
"""Summarize checkpointing-off against the completed checkpointed baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.monitoring_gradient_checkpointing.core import summarize
from experiments.monitoring_lr_sweep.prepare import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        json.loads(args.baseline.read_text()),
        json.loads(args.candidate.read_text()),
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
