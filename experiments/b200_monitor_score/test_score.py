"""Contract and mocked route tests without model loading or paid infrastructure."""

import asyncio
import copy
import json
import math
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from experiments.b200_monitor_score.run import score_command
from gleipnir.serving import monitor_score
from gleipnir.serving.monitor_score import (
    ENDPOINT,
    MonitorScoreRequest,
    classification_overrides,
    score_payload,
    validate_score_response,
)


def test_binary_margin_matches_two_way_softmax_and_large_logits():
    for logits in [[3.0, 4.5], [-4.5, -3.0], [10000.0, 9998.0]]:
        result = score_payload(logits, 20)
        assert result["score"] == pytest.approx(1 / (1 + math.exp(-result["margin"])))
    assert score_payload([0.0, -10000.0], 1)["score"] == 0
    assert score_payload([-10000.0, 0.0], 1)["score"] == 1
    assert score_payload([2.0, 2.0], 1)["score"] == 0.5


@pytest.mark.parametrize(
    "logits", [[1.0], [1.0, 2.0, 3.0], [math.nan, 1], [1, math.inf]]
)
def test_invalid_logits_fail_closed(logits):
    with pytest.raises(ValueError, match="finite decision"):
        score_payload(logits, 10)


def test_response_rejects_truncation_and_inconsistent_score():
    payload = score_payload([1.0, 2.0], 10)
    assert validate_score_response(payload, 10)["margin"] == 1
    with pytest.raises(ValueError, match="token mismatch"):
        validate_score_response(payload, 11)
    with pytest.raises(ValueError, match="inconsistent"):
        validate_score_response({**payload, "score": 0.2}, 10)
    with pytest.raises(ValueError, match="nonempty"):
        score_payload([1.0, 2.0], 0)


@pytest.mark.parametrize(
    "payload", [{"prompt": ""}, {"prompt": [15]}, {"prompt": "x", "max_tokens": 1}]
)
def test_request_rejects_generation_and_empty_or_token_inputs(payload):
    with pytest.raises(ValidationError):
        MonitorScoreRequest(**payload)


def test_conversion_preserves_nested_backbone_and_bf16_head():
    config = {
        "text_config": {
            "hidden_size": 2560,
            "num_hidden_layers": 32,
            "nested": {"a": 1},
        }
    }
    original = copy.deepcopy(config)
    overrides = classification_overrides(config)
    assert config == original
    assert overrides["text_config"]["hidden_size"] == 2560
    assert overrides["text_config"]["nested"] == {"a": 1}
    assert overrides["text_config"]["classifier_from_token"] == ["0", "1"]
    assert overrides["head_dtype"] == "model" and overrides["is_causal"]


def test_command_keeps_cached_batching_and_replaces_generation():
    parent = [
        "python",
        "-m",
        "old.server",
        "--worker-cls",
        "old.Worker",
        "--additional-config",
        json.dumps({"gleipnir_frost_fp4": {}, "serving_condition": {}}),
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        "32768",
        "--logprobs-mode",
        "processed_logprobs",
    ]
    old = parent.copy()
    new = score_command(
        parent, {"text_config": {"hidden_size": 2560}}, {"helper.py": "sha"}
    )
    assert parent == old
    assert new[new.index("--runner") + 1] == "pooling"
    assert new[new.index("--convert") + 1] == "classify"
    assert new[new.index("--max-num-batched-tokens") + 1] == "32768"
    assert "--logprobs-mode" not in new
    pooler = json.loads(new[new.index("--pooler-config") + 1])
    assert pooler == {
        "task": "classify",
        "pooling_type": "LAST",
        "use_activation": False,
    }
    with pytest.raises(ValueError, match="cached parent"):
        score_command([x for x in parent if x != "--enable-chunked-prefill"], {}, {})


def test_route_uses_raw_classification_without_special_tokens(monkeypatch):
    calls = []
    factories = ModuleType("vllm.entrypoints.pooling.factories")
    factories.init_pooling_state = lambda engine, state, args, logger, tasks: setattr(
        state, "serving_classification", SimpleNamespace(chat_template_config=None)
    )
    factories.register_pooling_api_routers = lambda *args: None

    class Serving:
        def __init__(self, engine, models, **kwargs):
            pass

        async def __call__(self, request, raw_request):
            calls.append(request)
            if request.model != "monitor":
                raise NotFound("unknown model")
            result = SimpleNamespace(
                outputs=SimpleNamespace(
                    data=SimpleNamespace(tolist=lambda: [2.0, 3.0])
                ),
                prompt_token_ids=[1, 2, 3],
            )
            return self._build_response(SimpleNamespace(final_res_batch=[result]))

    class NotFound(Exception):
        pass

    async def validate_json():
        return None

    modules = {
        "vllm.entrypoints.pooling": {"factories": factories},
        "vllm.entrypoints.pooling.classify.protocol": {
            "ClassificationCompletionRequest": SimpleNamespace
        },
        "vllm.entrypoints.pooling.classify.serving": {"ServingClassification": Serving},
        "vllm.entrypoints.serve.utils.api_utils": {
            "load_aware_call": lambda fn: fn,
            "with_cancellation": lambda fn: fn,
            "validate_json_request": validate_json,
        },
        "vllm.exceptions": {"VLLMNotFoundError": NotFound},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        monitor_score.importlib.metadata, "version", lambda name: "0.24.0"
    )
    monitor_score.install_score_api()
    app = FastAPI()
    app.state.openai_serving_models = None
    factories.init_pooling_state(None, app.state, None, None, ("classify",))
    factories.register_pooling_api_routers(app, ("classify",))

    async def requests():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as client:
            response = await client.post(ENDPOINT, json={"prompt": "already rendered"})
            assert response.status_code == 200
            assert response.json() == score_payload([2.0, 3.0], 3)
            missing = await client.post(
                ENDPOINT, json={"prompt": "x", "model": "unknown"}
            )
            assert missing.status_code == 404
            invalid = await client.post(
                ENDPOINT, json={"prompt": "x", "temperature": 0}
            )
            assert invalid.status_code == 422

    asyncio.run(requests())
    assert calls[0].input == "already rendered"
    assert not calls[0].add_special_tokens and not calls[0].use_activation
