"""Sequential independent 32/64-update arms with automatic merge and ID scoring."""

import json
from pathlib import Path

import yaml

from experiments.native_fp8_qat.run import main as run_arm
from experiments.qad_budget.prepare import prepare
from gleipnir.data.monitoring import write_json


def main():
    root = Path.cwd()
    config = yaml.safe_load((root / "experiments/qad_budget/config.yaml").read_text())
    output = root / "results" / config["experiment"]
    try:
        write_json(output / "status.json", {"phase": "preparing"})
        paths = prepare(root, config)
        for updates, path in zip((32, 64), paths, strict=True):
            write_json(
                output / "status.json", {"phase": "running", "arm_updates": updates}
            )
            run_arm(str(path.relative_to(root)))
        write_json(
            output / "summary.json",
            {
                "arms": [
                    json.loads(
                        (
                            root / f"results/qad-budget{n}-01/quality_comparison.json"
                        ).read_text()
                    )
                    for n in (32, 64)
                ],
                "promotion": False,
            },
        )
        write_json(output / "status.json", {"phase": "complete"})
    except Exception as exc:
        write_json(
            output / "status.json",
            {"phase": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise


if __name__ == "__main__":
    main()
