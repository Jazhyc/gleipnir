"""Run the shared summarize implementation for CoT-only removal."""

from pathlib import Path

from experiments.evaluation.id_action_only_evaluation.summarize import main

if __name__ == "__main__":
    main(
        config_path=Path("experiments/evaluation/id_cot_only_evaluation/config.json"),
        output_root=Path("results/id_cot_only_evaluation"),
    )
