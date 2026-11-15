import json

import pytest

from experiments.serving_concurrency import lengths, run, summarize


def test_levels_sweep_only_at_both_ends():
    assert run.levels("bf16") == run.SWEEP
    assert run.levels("fast") == run.SWEEP
    assert run.levels("quant") == (128, 1)
    assert run.levels("fast", ends=True) == (128, 1)
    assert run.levels("fast", top=64) == (64, 32, 16, 8, 4, 2, 1)
    assert run.levels("quant", top=64) == (64, 16, 1)
    assert run.levels("fast_g2048") == run.SWEEP
    assert run.levels("quant_g2048") == (128, 1)


def test_graph_helpers_change_only_the_graph_sizes():
    sizes = json.dumps({"cudagraph_capture_sizes": [1, 1280, 1536, 2048]})
    command = ["python", "--port", "1", run.GRAPHS, sizes, "--seed", "0"]
    assert run.without_graphs(command) == ["python", "--port", "1", "--seed", "0"]
    assert run.graph_sizes(command, 2048) == [1, 1280, 1535, 1536, 2048]
    assert run.graph_sizes(command, 1024) == [1]
    assert run.graph_sizes(command, 8192)[5:] == [
        *(2560, 3072, 3584, 4096, 5120, 6144, 7168, 8191, 8192)
    ]
    with pytest.raises(ValueError):
        run.graph_sizes(command, 3000)
    with pytest.raises(ValueError):
        run.graph_sizes(["python", run.GRAPHS, '{"cudagraph_capture_sizes": [1]}'], 512)
    assert run.graph_step("fast_g2048") == ("fast", 2048, "")
    assert run.graph_step("fast_g2048d") == ("fast", 2048, "d")
    assert 288 in run.graph_sizes(command, 2048, "o")
    assert 752 in run.graph_sizes(command, 2048, "d")
    assert run.graph_step("fast") is None


def test_swap_worker_changes_only_the_worker():
    command = ["python", "--worker-cls", "a.B", "--port", "1"]
    swapped = run.swap_worker(command, run.NO_TRIM)
    assert swapped == ["python", "--worker-cls", run.NO_TRIM, "--port", "1"]
    assert command[2] == "a.B"


def test_prepare_rejects_unknown_step(tmp_path):
    with pytest.raises(ValueError):
        run.prepare("27b", "route", tmp_path)


def test_percentile_interpolates():
    assert summarize.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert summarize.percentile([5.0], 0.95) == 5.0


def test_condition_summary_takes_medians(tmp_path):
    for repeat, seconds in enumerate((1.0, 2.0, 4.0)):
        rows = [
            {"latency_seconds": 0.1 * (repeat + 1), "prompt_tokens": 100},
            {"latency_seconds": 0.3 * (repeat + 1), "prompt_tokens": 300},
        ]
        (tmp_path / f"full_c4_repeat{repeat}.json").write_text(
            json.dumps({"rows": rows, "seconds": seconds})
        )
    (tmp_path / "condition.json").write_text("{}")
    item = summarize.condition_summary(tmp_path)["full_c4"]
    assert item["passes"] == 3
    assert item["prompt_tokens_per_second"] == 200.0
    assert item["p50_ms"] == pytest.approx(400.0)


def command(limit: str = "32768") -> list[str]:
    overrides = {
        "text_config": {
            "max_position_embeddings": 262144,
            "rope_parameters": {"rope_type": "default"},
        }
    }
    return [
        "python",
        "--max-model-len",
        limit,
        "--max-num-batched-tokens",
        "32768",
        "--hf-overrides",
        json.dumps(overrides),
    ]


def test_native_phase_changes_only_the_limit():
    original = command()
    changed = lengths.phase_command(original, "native")
    assert changed[2] == "262144"
    assert changed[:2] + changed[3:] == original[:2] + original[3:]


def test_yarn_phase_scales_rope():
    changed = lengths.phase_command(command("262144"), "yarn")
    text = json.loads(changed[-1])["text_config"]
    assert changed[2] == "1048576"
    assert text["max_position_embeddings"] == 1048576
    assert text["rope_parameters"]["rope_type"] == "yarn"
    assert text["rope_parameters"]["factor"] == 4.0


def test_phase_command_rejects_other_chunk_budget():
    bad = command()
    bad[4] = "16384"
    with pytest.raises(ValueError):
        lengths.phase_command(bad, "native")


def test_agent_only_keeps_tool_trajectories_in_order():
    rows = [
        {"id": 1, "dataset": "tool_trajectory/stride"},
        {"id": 2, "dataset": "liars-bench/soft-trigger"},
        {"id": 3, "dataset": "tool_trajectory/gloom"},
        {"id": 4, "dataset": "tool_trajectory/cot_red_handed"},
    ]
    assert [row["id"] for row in run.agent_only(rows)] == [1, 3]
