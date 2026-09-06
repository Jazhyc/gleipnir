"""Resolve the pretrained-only ID arms and reuse the direct-logit evaluator."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from experiments.tool_trajectory_monitoring.benchmark_qwen_ood import (
    run,
    validate_config,
)
from experiments.tool_trajectory_monitoring.qwen_reasoning_core import (
    atomic_write_json,
    load_json,
    load_jsonl,
)
from gleipnir.calibration import binary_calibration


def resolve_config(path: Path, size: str) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    config["model"] = config.pop("models")[size]
    config["campaign_id"] += f"-{size}"
    validate_config(config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=["4b", "9b"], required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/pretrained_id/config.yaml")
    )
    args = parser.parse_args()
    output = Path("results/pretrained_id") / args.size
    config_path = output / "config.json"
    config = resolve_config(args.config, args.size)
    if config_path.exists() and load_json(config_path) != config:
        raise ValueError(
            "refusing to overwrite a different resolved evaluation contract"
        )
    atomic_write_json(config_path, config)
    run(
        argparse.Namespace(
            config=config_path, output=output, force=False, stop_after_canary=False
        )
    )
    rows = load_jsonl(output / "predictions.jsonl")
    views = {"pooled": rows}
    views.update(
        {
            source: [r for r in rows if r["source"] == source]
            for source in config["scope"]["sources"]
        }
    )
    calibration = {
        name: {
            str(n): binary_calibration(
                [r["label"] for r in subset], [r["score"] for r in subset], n
            )
            for n in (5, 10, 20)
        }
        for name, subset in views.items()
    }
    atomic_write_json(output / "calibration.json", calibration)


if __name__ == "__main__":
    main()
