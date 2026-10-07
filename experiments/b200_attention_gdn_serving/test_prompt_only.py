"""The offline path must reject continuation and partial prompt histories."""

import json

import pytest

from experiments.b200_attention_gdn_serving.run import EXPERIMENT, resolve_condition
from experiments.b200_inference_benchmark.run import server_command
from gleipnir.serving_prompt_only_contract import (
    validate_complete_prompts,
    validate_prompt_only_request,
)


@pytest.mark.parametrize(
    "lengths,computed",
    [
        ([9], [1]),
        ([0], [0]),
        ([32769], [0]),
        ([20000, 20000], [0, 0]),
        ([1] * 129, [0] * 129),
        ([9], []),
    ],
)
def test_reject_partial_or_unsupported_batch(lengths, computed):
    with pytest.raises(ValueError):
        validate_complete_prompts(lengths, computed)


def test_whole_prompts_and_single_decision():
    validate_complete_prompts([188, 28733], [0, 0])
    validate_prompt_only_request(1)


@pytest.mark.parametrize("tokens,n", [(None, 1), (2, 1), (1, 2), (0, 1)])
def test_reject_continuation(tokens, n):
    with pytest.raises(ValueError):
        validate_prompt_only_request(tokens, n)


def test_recipe_requires_whole_prompts_and_native_admission():
    recipe = json.loads((EXPERIMENT / "prompt_only.json").read_text())
    resolved = resolve_condition(recipe, {})
    assert resolved["startup_audit"].endswith("native_prompt_only.json")
    assert "--compilation-config" in resolved["extra_server_args"]
    with pytest.raises(ValueError, match="prompt-only worker"):
        resolve_condition({**recipe, "prompt_only": False}, {})
    for key, value in (("prompt_only_validation", None), ("compilation_config", {})):
        with pytest.raises(ValueError, match="admitted whole-prompt"):
            resolve_condition({**recipe, key: value}, {})
    recipe["serving_config_overrides"]["enable_chunked_prefill"] = True
    with pytest.raises(ValueError, match="admitted whole-prompt"):
        resolve_condition(recipe, {})


def test_command_disables_chunking_only_when_requested():
    config = dict(
        model="m",
        revision="r",
        port=1,
        adapter="a",
        max_model_len=32768,
        max_num_seqs=128,
        max_num_batched_tokens=32768,
        gpu_memory_utilization=0.9,
        seed=0,
    )
    assert "--enable-chunked-prefill" in server_command(config)
    assert "--no-enable-chunked-prefill" in server_command(
        {**config, "enable_chunked_prefill": False}
    )
