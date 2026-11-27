import json

import pytest

from experiments.serving.gleipnir4b_serving_ladder import compare, run


def rows(scores):
    return [{"id": i, "prompt_tokens": 1000, "score": s} for i, s in enumerate(scores)]


def test_cost_per_thousand():
    # The recorded release row: 226K tokens/s at $7.99/h is $0.0850 per 1,000.
    assert compare.cost_per_thousand(226_000, 7.99) == pytest.approx(0.0850, abs=2e-4)
    with pytest.raises(ValueError):
        compare.cost_per_thousand(0, 7.99)


def test_summarize_reports_speedups_and_score_drift():
    table = compare.summarize(
        {
            "bf16": [(rows([0.1, 0.9]), 2.0), (rows([0.1, 0.9]), 2.0)],
            "fast": [(rows([0.2, 0.9]), 1.0)],
        }
    )
    assert table["bf16"]["median_prompt_tokens_per_second"] == 1000
    assert table["fast"]["speedup_over_first"] == 2
    assert table["bf16"]["speedup_over_previous"] is None
    assert table["bf16"]["score_mae_against_last"] == pytest.approx(0.05)
    assert (
        table["fast"]["usd_per_1000"]["committed"]
        < (table["bf16"]["usd_per_1000"]["committed"])
    )


def test_retarget_rejects_other_differences():
    def command(python, model, worker, config, x):
        return [python, "--model", model, "--host", "h", "--port", "1"] + [
            "--worker-cls",
            worker,
            "--additional-config",
            json.dumps(config),
            "--x",
            x,
        ]

    release = command("py", "m", "A", {"serving_condition": {"merged_model": "m"}}, "1")
    layered = command(
        "old", "o", "B", {"serving_condition": {"merged_model": "o"}, "k": 1}, "1"
    )
    out = run.retarget(layered, release)
    assert out[:7] == release[:7]
    assert json.loads(out[10]) == {"serving_condition": {"merged_model": "m"}, "k": 1}
    layered[-1] = "2"
    with pytest.raises(ValueError):
        run.retarget(layered, release)
