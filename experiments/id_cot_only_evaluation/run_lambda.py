"""Run the shared run_lambda implementation for CoT-only removal."""

from pathlib import Path

from experiments.id_action_only_evaluation.run_lambda import main

if __name__ == "__main__":
    main(
        config_path=Path("experiments/id_cot_only_evaluation/config.json"),
        output_root=Path("results/id_cot_only_evaluation"),
        log_root=Path("logs/lambda/id_cot_only_evaluation"),
    )
