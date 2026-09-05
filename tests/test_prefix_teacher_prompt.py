from pathlib import Path


def test_prefix_prompt_preserves_original_rubric():
    root = Path(__file__).resolve().parents[1]
    original = (
        root / "experiments/tool_trajectory_monitoring/prompts/teacher.txt"
    ).read_text()
    prefix = (
        root / "experiments/monitoring_prefix_supervision/teacher_prefix.txt"
    ).read_text()
    before, added = prefix.split("TRAJECTORY SO FAR\n\n", 1)
    temporal_scope, after = added.split("SECURITY BOUNDARY", 1)
    restored = before + "SECURITY BOUNDARY" + after
    restored = restored.replace(
        "any subsequent actions visible within this prefix",
        "what the agent subsequently does",
    ).replace("and any report visible so far.", "and the final report.")
    restored = restored.replace(
        "using the entire visible trajectory prefix.", "using the full trajectory."
    )
    assert restored == original
    assert "Do not predict what the agent will do later" in temporal_scope
    assert "not evidence that the agent has finished its task" in temporal_scope
