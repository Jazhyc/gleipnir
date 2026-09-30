import pytest

from gleipnir import attention_backends as ab


def test_eager_interface_wraps_only_the_requested_registered_backend(
    monkeypatch,
) -> None:
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    seen = []

    def sentinel(*args, **kwargs):
        return None

    monkeypatch.setattr(
        torch.compiler,
        "disable",
        lambda function: sentinel if function is original else None,
    )
    monkeypatch.setattr(
        ALL_ATTENTION_FUNCTIONS,
        "register",
        lambda name, function: seen.append((name, function)),
    )
    result = ab.install_eager_attention_interface("sdpa")
    assert seen == [("sdpa", sentinel)]
    assert result["module"] == original.__module__
    with pytest.raises(ValueError):
        ab.install_eager_attention_interface(None)


def test_attention_selection_requires_installed_pinned_fa4(monkeypatch) -> None:
    monkeypatch.setattr(ab, "version", lambda _: "4.0.0b33")
    assert ab.attention_loader_kwargs(None) == {}
    assert ab.attention_loader_kwargs("sdpa") == {"attn_implementation": "sdpa"}
    assert ab.attention_loader_kwargs("flash_attention_4", "4.0.0b33") == {
        "attn_implementation": "flash_attention_4"
    }
    for backend, version in (
        ("flash_attention_4", None),
        ("flash_attention_4", "bad"),
        ("sdpa", "bad"),
        (None, "bad"),
        ("unknown", None),
    ):
        with pytest.raises(ValueError):
            ab.attention_loader_kwargs(backend, version)


def test_backend_canary_restores_candidate_even_on_forward_failure() -> None:
    class Model:
        backend = "flash_attention_4"

        def set_attn_implementation(self, backend):
            self.backend = backend

    model = Model()
    seen = []

    def forward():
        seen.append(model.backend)
        return model.backend

    result = ab.compare_attention_backends(
        model,
        reference="sdpa",
        candidate="flash_attention_4",
        forward=forward,
        compare=lambda ref, cand: {
            "passed": ref == "sdpa" and cand == "flash_attention_4"
        },
    )
    assert result["passed"] and seen == ["sdpa", "flash_attention_4"]
    assert model.backend == "flash_attention_4"

    def failing_forward():
        raise RuntimeError("canary failed")

    with pytest.raises(RuntimeError):
        ab.compare_attention_backends(
            model,
            reference="sdpa",
            candidate="flash_attention_4",
            forward=failing_forward,
            compare=lambda *_: {},
        )
    assert model.backend == "flash_attention_4"
