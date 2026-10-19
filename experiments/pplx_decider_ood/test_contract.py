import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.pplx_decider_ood.run import load_question, stable_sha256
from experiments.pplx_decider_ood.summarize import ranking

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "experiments/pplx_decider_ood/config.json").read_text())


def test_question_is_the_frozen_jev_noul_question() -> None:
    key, question = load_question(CONFIG)
    assert key == "problematic_behavior"
    assert question["type"] == "noul"
    assert (
        stable_sha256(question)
        == "9b423177a4d48897d083f927845ab464fe0f97e758f64159aedee94b3dab1f12"
    )


def test_inputs_match_the_jev_frozen_states() -> None:
    assert CONFIG["inputs"]["full"]["sha256"] == (
        "c38c6214577f0a4e68ab20ba5531d94da8bdaa26efc1420b0c77ef2ccdd2b99a"
    )
    assert CONFIG["inputs"]["full"]["rows"] == 6395


def test_list_prices_are_input_only() -> None:
    models = CONFIG["models"]
    assert models["flash"]["input_price_usd_per_million"] == 0.01
    assert models["decider"]["input_price_usd_per_million"] == 0.02
    assert all(m["output_price_usd_per_million"] == 0 for m in models.values())


def test_ranking_uses_margin_and_skips_single_label_sources() -> None:
    frame = pd.DataFrame(
        {
            "source": ["a"] * 4 + ["b"] * 2,
            "label": [0, 0, 1, 1, 1, 1],
            # Saturated probabilities would tie; margins still separate.
            "margin": [-3.0, 50.0, 60.0, 70.0, 1.0, 2.0],
        }
    )
    result = ranking(frame)
    assert result["by_source"]["b"] is None
    assert np.isclose(result["mean_auroc"], 1.0)
    assert np.isclose(result["mean_pauroc_at_20"], 1.0)


def test_teacher_question_is_the_specification_with_mechanical_edits() -> None:
    from experiments.pplx_decider_ood.prepare_teacher_question import SPEC, build

    built = build()
    _, question = load_question(CONFIG, "teacher")
    assert built["question"] == question
    spec = SPEC.read_text()
    assert "Prediction" not in question["instructions"]
    assert "OUTPUT INTERFACE" not in question["instructions"]
    assert question["criteria"]["true"].removeprefix("The trajectory contains ") in spec
    assert question["criteria"]["false"].removeprefix("The visible actions ") in spec
    assert "SILENT REVIEW PROCEDURE" in question["instructions"]
