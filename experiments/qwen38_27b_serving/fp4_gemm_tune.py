"""Run the 4B FP4 tile sweep at a served model's projection shapes.

Sets the 4B tuning module's ``SHAPES`` to ``fp4_shapes`` of ``--model`` for
this process only, then runs the unchanged sweep
(`experiments/b200_attention_gdn_serving/fp4_gemm_tune.py`: 6 tiles x 8 row
counts per shape, graph replay, changed-input replay, 2% selection gain). The
receipt additionally records the shapes and this wrapper's sources.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from experiments.b200_attention_gdn_serving import fp4_gemm_tune
    from gleipnir.data.monitoring import file_hash
    from gleipnir.serving.fp4 import tuning
    from gleipnir.serving.fp4_geometry_tuning import fp4_shapes

    config = json.loads((args.model / "config.json").read_text())
    shapes = fp4_shapes(SimpleNamespace(**config.get("text_config", config)))
    tuning.SHAPES = shapes
    sys.argv = [fp4_gemm_tune.__file__, "--output", str(args.output)]
    fp4_gemm_tune.main()
    report = json.loads(args.output.read_text())
    report["shapes"] = {name: list(dims) for name, dims in shapes.items()}
    report["model_config_sha256"] = file_hash(args.model / "config.json")
    for path in (Path(__file__), Path("src/gleipnir/serving/fp4_geometry_tuning.py")):
        relative = str(path.resolve().relative_to(Path.cwd()))
        report["sources"][relative] = file_hash(path)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("fp4_gemm_tune_selected", report["selected"], flush=True)


if __name__ == "__main__":
    main()
