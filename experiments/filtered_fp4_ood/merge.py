"""Run FP32-accumulating reconstruction in the pinned training runtime."""

from pathlib import Path

import yaml

from experiments.filtered_fp4_ood.run import restore_merge

if __name__ == "__main__":
    root = Path.cwd()
    config = yaml.safe_load(
        (root / "experiments/filtered_fp4_ood/config.yaml").read_text()
    )
    restore_merge(root, config, root / "results" / config["experiment"])
