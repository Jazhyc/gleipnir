#!/usr/bin/env python3
"""Run on the existing Pod to stage serving dependencies on ephemeral storage."""

import argparse
import json
from pathlib import Path

from gleipnir.serving_runtime import DEFAULT_RUNTIME, stage_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_RUNTIME)
    args = parser.parse_args()
    receipt = stage_runtime(args.root, args.target)
    print(
        json.dumps({k: receipt[k] for k in ["python", "staging_seconds", "ephemeral"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
