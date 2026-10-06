"""Graph padding cannot change precision bands or the serving contract."""

import asyncio
import copy
import json
from pathlib import Path

import pytest

from experiments.b200_attention_gdn_serving.run import resolve_condition
from gleipnir.serving_prefill_graphs import graph_padding_allowed, validate_graph_config


def recipe():
    return json.loads((Path(__file__).parent / "prefill_graphs.json").read_text())


@pytest.mark.parametrize(
    "rows,padded,allowed",
    [
        (127, 128, True),
        (1023, 1024, True),
        (1535, 1536, False),
        (1536, 1536, True),
        (2047, 2048, True),
        (4096, 4096, True),
        (4096, 8192, False),
        (4097, 8192, False),
        (29116, 32768, False),
        (29128, 32768, True),
        (32760, 32768, True),
        (0, 1, False),
        (512, 256, False),
    ],
)
def test_bounded_padding_preserves_producer(rows, padded, allowed):
    assert graph_padding_allowed(rows, padded, 0.125) is allowed


def test_graph_recipe_preserves_reference():
    condition = recipe()
    args = resolve_condition(condition, {})["extra_server_args"]
    assert json.loads(
        args[args.index("--compilation-config") + 1]
    ) == validate_graph_config(condition)
    assert condition["baseline"] == condition["high_reference"] == "selected"
    assert "gdn_direct_output_validation" not in condition
    changed = {
        **condition,
        "worker_cls": condition["worker_cls"].replace(
            "prefill_graph_worker", "swiglu_native_output_worker"
        ),
    }
    with pytest.raises(ValueError, match="bounded audited worker"):
        resolve_condition(changed, {})


@pytest.mark.parametrize(
    "key,value",
    [
        ("cudagraph_mode", "FULL"),
        ("mode", "NONE"),
        ("cudagraph_capture_sizes", [1, 32769]),
        ("cudagraph_capture_sizes", [1, 1024]),
        ("cudagraph_capture_sizes", [32768, 1]),
        ("cudagraph_capture_sizes", [1, 1, 32768]),
    ],
)
def test_reject_unsupported_capture(key, value):
    c = copy.deepcopy(recipe())
    c["compilation_config"][key] = value
    with pytest.raises(ValueError, match="unsupported prefill graph"):
        validate_graph_config(c)


def test_reject_excessive_padding():
    c = recipe()
    c["prefill_graphs"]["max_padding_fraction"] = 0.5
    with pytest.raises(ValueError, match="unsupported prefill graph"):
        validate_graph_config(c)


@pytest.mark.parametrize("fail_request", [False, True])
def test_canary_restores_replay_and_checks_changed_logprobs(
    tmp_path, monkeypatch, fail_request
):
    from experiments.b200_attention_gdn_serving import prefill_graph_canary as module

    switches, calls = [], []

    async def state(client, enabled):
        switches.append(enabled)
        return {
            "decisions": [
                {
                    "mode": "PIECEWISE",
                    "rows": 32760,
                    "padded": 32768,
                    "reason": "upstream",
                },
                {"mode": "NONE", "rows": 127, "padded": 127, "reason": "canary_bypass"},
                {
                    "mode": "NONE",
                    "rows": 4097,
                    "padded": 4097,
                    "reason": "padding_or_precision_band",
                },
            ]
        }

    async def measured(client, rows, ids, concurrency, model):
        assert model == "base" and concurrency == 1
        calls.append(rows)
        if fail_request:
            raise RuntimeError("request failed")
        return [dict(score=0.5, logprob_0=-0.7, logprob_1=-0.7) for r in rows], 1

    monkeypatch.setattr(module, "graph_state", state)
    monkeypatch.setattr(module, "trial", measured)
    if fail_request:
        with pytest.raises(RuntimeError, match="request failed"):
            asyncio.run(module.canary(None, [15, 16], tmp_path))
        assert switches == ["false", "true"]
    else:
        report = asyncio.run(module.canary(None, [15, 16], tmp_path))
        assert report["passed"] and report["rows"] == 24
        assert switches == ["false", "true", "true"]
        assert len({r["prompt_sha256"] for r in calls[0]}) == 24
