"""Replay prompts share exact prefixes and the command changes only caching."""

import pytest

from experiments.b200_prefix_replay.run import enable_prefix_caching
from experiments.b200_prefix_replay.session import (
    SENTINEL,
    TAIL,
    content,
    policies,
    render,
)


def session():
    steps = [
        {
            "block": "[ASSISTANT]\n[TOOL CALLS]\nbash({})\n\n[TOOL]\nout1\n\n",
            "action": "[ASSISTANT]\n[TOOL CALLS]\nbash({})\n",
        },
        {"block": "[ASSISTANT]\nthinking only\n\n", "action": None},
        {
            "block": "[ASSISTANT]\n[TOOL CALLS]\nls({})\n\n[TOOL]\nout2\n\n",
            "action": "[ASSISTANT]\n[TOOL CALLS]\nls({})\n",
        },
    ]
    return {
        "header": "H<agent_trajectory>\n[USER]\nu\n",
        "steps": steps,
        "template": f"<u>{SENTINEL}</u>Prediction:",
    }


def test_checks_extend_previous_history_and_end_with_the_action():
    s = session()
    first, second = content(s, 0), content(s, 2)
    assert first.endswith("bash({})\n" + TAIL)
    shared = first[: -len(TAIL)]
    assert second.startswith(shared)
    assert render(s, 2).startswith("<u>" + shared)
    assert policies(s["steps"])["every_action"] == [0, 2]


def test_command_changes_only_prefix_policy():
    command = ["python", "--no-enable-prefix-caching", "--seed", "0"]
    changed = enable_prefix_caching(command)
    assert changed == [
        "python",
        "--enable-prefix-caching",
        "--seed",
        "0",
        "--mamba-cache-mode",
        "align",
    ]
    with pytest.raises(ValueError):
        enable_prefix_caching(changed)


def test_opt_in_serving_changes_only_cache_flags_and_worker():
    import json

    from experiments.b200_prefix_replay.serving import (
        CACHED_WORKER,
        DEFAULT_WORKER,
        WORKER_SOURCE,
        prefix_cached_command,
    )

    additional = json.dumps({"gleipnir_frost_fp4": {"a.py": "x"}})
    command = [
        "python",
        "--worker-cls",
        DEFAULT_WORKER,
        "--no-enable-prefix-caching",
        "--additional-config",
        additional,
    ]
    changed = prefix_cached_command(command)
    assert changed[2] == CACHED_WORKER and "--enable-prefix-caching" in changed
    assert changed[-2:] == ["--mamba-cache-mode", "align"]
    bound = json.loads(changed[changed.index("--additional-config") + 1])
    assert bound["gleipnir_frost_fp4"]["a.py"] == "x"
    assert len(bound["gleipnir_frost_fp4"][WORKER_SOURCE]) == 64
    other = command.copy()
    other[2] = "some.OtherWorker"
    with pytest.raises(ValueError, match="default worker"):
        prefix_cached_command(other)
