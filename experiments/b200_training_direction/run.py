"""Compatibility entrypoint for the frozen small training-direction comparison."""

from pathlib import Path

from gleipnir.evaluation.direction_campaign import (
    replacement_command as replacement_command,
)


def main() -> None:
    from gleipnir.evaluation.direction_campaign import main as run_campaign

    run_campaign(default_config=Path(__file__).with_name("config.json"))


if __name__ == "__main__":
    main()
