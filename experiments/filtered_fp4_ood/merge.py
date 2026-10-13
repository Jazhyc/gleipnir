"""Run FP32-accumulating reconstruction in the pinned training runtime."""

from pathlib import Path

from experiments.filtered_fp4_ood.run import load_config, restore_merge

if __name__ == "__main__":
    root = Path.cwd()
    _, config = load_config()
    restore_merge(root, config, root / "results" / config["experiment"])
