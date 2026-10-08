"""CPU contract tests for request-specific residual steering and capture bounds."""

import pytest
import torch
from pydantic import ValidationError
from vllm_lens import SteeringVector

from gleipnir.serving.lens import LensScoreRequest, validate_intervention
from gleipnir.serving.lens_worker import (
    MonitorLensExtension,
    intervene_slice,
    residual_stream,
)


def test_capture_positions_are_unambiguous_and_strict():
    r = LensScoreRequest(prompt="x", capture_layers=[3, 0], capture_positions=[5, 2])
    assert r.capture_layers == [0, 3] and r.capture_positions == [2, 5]
    for value in ([0, 0], [-1], [True]):
        with pytest.raises(ValidationError):
            LensScoreRequest(prompt="x", capture_positions=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capture_layers": [32]},
        {"capture_positions": [8]},
        {"capture_layers": list(range(32)), "capture_positions": "all"},
    ],
)
def test_invalid_capture_fails_before_admission(kwargs):
    with pytest.raises(ValueError):
        validate_intervention(
            LensScoreRequest(prompt="x", **kwargs),
            layers=32,
            hidden=2560,
            tokens=32768 if kwargs.get("capture_positions") == "all" else 8,
        )


def test_norm_matching_uses_full_fused_residual_and_isolates_neighbors():
    delta, residual = torch.zeros(4, 3), torch.full((4, 3), 2.0)
    vector = SteeringVector(
        activations=torch.ones(1, 3), layer_indices=[3], norm_match=True, scale=0.5
    )
    changed = intervene_slice((delta, residual), [vector], 3, 1, 3, 0)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(changed[1], residual)
    stream = residual_stream(changed)
    assert torch.equal(stream[[0, 3]], residual[[0, 3]])
    torch.testing.assert_close(
        stream[1:3], torch.full((2, 3), 3.0), atol=1e-6, rtol=1e-6
    )


def test_absolute_steering_positions_cross_chunk_boundaries():
    vector = SteeringVector(
        activations=torch.ones(1, 1, 3),
        layer_indices=[3],
        position_indices=[7],
        scale=2,
    )
    early = intervene_slice(torch.zeros(4, 3), [vector], 3, 0, 4, 0)
    late = intervene_slice(torch.zeros(4, 3), [vector], 3, 0, 4, 4)
    assert not torch.count_nonzero(early)
    assert torch.equal(late[:3], torch.zeros(3, 3))
    assert torch.equal(late[3], torch.full((3,), 2.0))


def test_nonfinite_vectors_and_bad_position_geometry_are_rejected():
    for vector in [
        SteeringVector(activations=torch.full((1, 3), float("nan")), layer_indices=[0]),
        SteeringVector(
            activations=torch.ones(1, 3), layer_indices=[0], position_indices=[1]
        ),
        SteeringVector(
            activations=torch.ones(1, 2, 3), layer_indices=[0], position_indices=[1]
        ),
    ]:
        with pytest.raises(ValueError):
            validate_intervention(
                LensScoreRequest(
                    prompt="x", steering_vectors=[vector.model_dump(mode="json")]
                ),
                layers=4,
                hidden=3,
                tokens=8,
            )


def test_chunked_capture_and_steering_do_not_leak_between_pooling_requests(monkeypatch):
    from types import SimpleNamespace

    from vllm_lens._helpers._serialize import deserialize_tensor

    import gleipnir.serving.lens_worker as module

    vector = SteeringVector(
        activations=torch.ones(1, 1, 3),
        layer_indices=[3],
        position_indices=[7],
        scale=2,
    )
    configs = {
        "a": {
            "request_id": "a",
            "capture_layers": [3],
            "capture_positions": [0, 7],
            "steering_vectors": [vector.model_dump(mode="json")],
        },
        "b": {
            "request_id": "b",
            "capture_layers": [3],
            "capture_positions": "all",
            "steering_vectors": [],
        },
    }
    worker = MonitorLensExtension()
    worker._lens_buffers, worker._lens_vectors = {}, {}
    worker.model_runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=["a", "b"], num_reqs=2, num_computed_tokens_cpu=torch.tensor([0, 0])
        ),
        query_start_loc=SimpleNamespace(np=torch.tensor([0, 4, 8])),
        requests={
            key: SimpleNamespace(
                pooling_params=SimpleNamespace(extra_kwargs={"gleipnir_lens": cfg}),
                prompt_token_ids=[0] * (8 if key == "a" else 4),
            )
            for key, cfg in configs.items()
        },
    )
    monkeypatch.setattr(module, "is_forward_context_available", lambda: True)
    assert worker._lens_forward(3, torch.zeros(8, 3)) is not None
    b = worker.lens_collect("b")
    assert b["activation_positions"] == [0, 1, 2, 3]
    assert not torch.count_nonzero(
        deserialize_tensor(b["activations"]["residual_stream"])
    )
    worker.model_runner.input_batch = SimpleNamespace(
        req_ids=["a"], num_reqs=1, num_computed_tokens_cpu=torch.tensor([4])
    )
    worker.model_runner.query_start_loc.np = torch.tensor([0, 4])
    changed = worker._lens_forward(3, torch.zeros(4, 3))
    assert torch.equal(changed[3], torch.full((3,), 2.0))
    a = worker.lens_collect("a")
    assert a["activation_positions"] == [0, 7]
    assert a["capture_chunks"]["3"] == [[0, 4], [4, 8]]
    actual = deserialize_tensor(a["activations"]["residual_stream"])
    assert torch.equal(actual, torch.tensor([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]]))
    assert not worker._lens_buffers and not worker._lens_vectors
