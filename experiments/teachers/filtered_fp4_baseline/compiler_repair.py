"""Opt-in typed-coordinate repair and native MXFP8 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def install_repair() -> None:
    """Repair derived source without changing the frozen vendor/template files."""
    import gleipnir.serving.mxfp8 as implementation
    from gleipnir.serving.mxfp8_compile_compat import typed_kv_coordinates_source

    if getattr(implementation, "_typed_coordinate_repair", False):
        return
    original = implementation.hardware_exp2_source

    def source(value: str) -> str:
        return typed_kv_coordinates_source(original(value))

    implementation.hardware_exp2_source = source
    implementation._typed_coordinate_repair = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    install_repair()
    from experiments.b200_attention_gdn_serving.mxfp8_canary import main as canary

    sys.argv = [sys.argv[0], "--output", str(args.output)]
    try:
        canary()
    finally:
        if args.output.exists():
            receipt = json.loads(args.output.read_text())
            root = Path.cwd()
            for value in [
                Path(__file__).resolve(),
                root / "src/gleipnir/serving/mxfp8_compile_compat.py",
            ]:
                receipt["sources"][str(value)] = hashlib.sha256(
                    value.read_bytes()
                ).hexdigest()
            receipt["compiler_repair"] = "typed_int32_kv_coordinates"
            args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
