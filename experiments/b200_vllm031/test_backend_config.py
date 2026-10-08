"""Keep backend selection and the existing compiler identity protocol intact."""

from types import SimpleNamespace

import pytest

from experiments.b200_vllm031.backend_config import nonparallel_call, resolver_config
from gleipnir.serving.compile_cache import ServingCompileConfig


def test_resolver_sees_explicit_backend_without_changing_engine_hash(tmp_path):
    data = {
        "serving_condition": {},
        "gleipnir_frost_fp4": {},
        "gdn_prefill_backend": "flashinfer",
    }
    wrapped = ServingCompileConfig(data, tmp_path, {})
    before = wrapped.compute_hash()
    model = SimpleNamespace(hf_text_config=SimpleNamespace(linear_key_head_dim=128))
    config = SimpleNamespace(additional_config=wrapped, model_config=model)
    view = resolver_config(config)
    assert isinstance(view.additional_config, dict)
    assert view.additional_config["gdn_prefill_backend"] == "flashinfer"
    assert view.model_config is model
    assert config.additional_config is wrapped
    assert not isinstance(wrapped, dict)
    assert wrapped.compute_hash() == before


def test_plain_upstream_config_needs_no_bridge():
    config = SimpleNamespace(additional_config={"gdn_prefill_backend": "flashinfer"})
    assert resolver_config(config) is config


def test_nonparallel_call_preserves_inputs_and_explicit_route():
    tensor = object()
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return tensor

    assert nonparallel_call(original, tensor, backend="flashinfer") is tensor
    assert calls == [((tensor,), {"backend": "flashinfer", "use_cp": False})]
    for kwargs in ({"backend": "auto"}, {"backend": "flashinfer", "use_cp": True}):
        with pytest.raises(ValueError):
            nonparallel_call(original, tensor, **kwargs)
    assert len(calls) == 1
