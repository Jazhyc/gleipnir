"""Populate FLA autotune caches before the long-context model is resident."""

from __future__ import annotations

import argparse

from gleipnir.qwen35_fast_training import prewarm_gated_delta_rule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, required=True)
    args = parser.parse_args()
    prewarm_gated_delta_rule(args.sequence_length)


if __name__ == "__main__":
    main()
