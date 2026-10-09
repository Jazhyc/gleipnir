"""Run the frozen concept census using the existing binary-logit audit runner."""

from pathlib import Path

from gleipnir.evaluation.concept_census import run

if __name__ == "__main__":
    run(Path("experiments/training_firewall_census/config.json"))
