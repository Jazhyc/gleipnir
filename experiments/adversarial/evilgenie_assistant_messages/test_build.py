import numpy as np

from experiments.adversarial.adversarial_natural_cheats.trajectories import render
from experiments.adversarial.evilgenie_assistant_messages import analyze, build


def messages(text: str) -> list[dict]:
    return [
        {"role": "user", "text": "Solve it."},
        {
            "role": "assistant",
            "text": text,
            "tool_calls": [{"function": "safe_bash", "arguments": {"cmd": "ls"}}],
        },
        {"role": "tool", "text": "solution.py", "function": "safe_bash", "error": None},
        {"role": "assistant", "text": text, "tool_calls": []},
    ]


def test_without_text_matches_baseline_render():
    m = messages("  \n")
    assert build.render_with_text(m) == render(m)
    assert not build.has_text(m)


def test_text_precedes_calls_as_in_training_data():
    out = build.render_with_text(messages("I will look."))
    assert "[ASSISTANT]\nI will look.\n\n[TOOL CALLS]\nsafe_bash(" in out
    assert out.endswith("[TOOL]\nsolution.py\n\n[ASSISTANT]\nI will look.")


def test_paired_difference_is_zero_for_identical_scores():
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([0.9, 0.1, 0.4, 0.5, 0.8, 0.2])
    g = np.array(["a", "a", "b", "b", "c", "c"])
    out = analyze.paired(y, s, s, g, draws=50)
    assert out["auroc"]["difference"] == 0
    assert out["auroc"]["difference_interval"] == [0.0, 0.0]
