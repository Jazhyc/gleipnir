"""Clamp semantics on fused residuals, final positions, and input validation."""

import pytest
import torch

from gleipnir.serving.lens import LensScoreRequest, validate_intervention
from gleipnir.serving.lens_projection import apply_projection


def edit(**kwargs):
    return (
        dict(
            direction=[1.0, 0.0, 0.0],
            layer_indices=[0],
            beta=0.25,
            decision_centers=[1.0],
            span_centers=[2.0],
        )
        | kwargs
    )


def test_projection_matches_reference_and_keeps_neighbors():
    delta = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 10
    residual = torch.ones_like(delta) * 2
    before = delta.clone()
    h = delta + residual
    d = torch.tensor([1.0, 0.0, 0.0])
    apply_projection(delta, h, d, 0.25, 1.0, 2.0, 1, 5, 4, 7)
    centers = torch.tensor([2.0, 2.0, 2.0, 1.0])
    expected = h[1:5] - 0.25 * ((h[1:5] @ d) - centers)[:, None] * d
    torch.testing.assert_close((delta + residual)[1:5], expected)
    assert torch.equal(delta[[0, 5]], before[[0, 5]])


def test_chunk_end_is_not_decision_position_and_zero_is_exact():
    delta = torch.zeros(3, 3, dtype=torch.bfloat16)
    h = torch.ones_like(delta) * 4
    d = torch.tensor([1.0, 0.0, 0.0])
    apply_projection(delta, h, d, 1.0, 1.0, 2.0, 0, 3, 0, 7)
    assert torch.equal(delta[:, 0], torch.full((3,), -2.0, dtype=delta.dtype))
    before = delta.clone()
    apply_projection(delta, h, d, 0.0, 1.0, 2.0, 0, 3, 0, 7)
    assert torch.equal(delta, before)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"direction": [2.0, 0.0, 0.0]},
        {"beta": float("nan")},
        {"beta": 1.1},
        {"span_centers": []},
        {"decision_centers": [float("inf")]},
        {"layer_indices": [1]},
    ],
)
def test_invalid_clamp_is_rejected_before_admission(kwargs):
    with pytest.raises(ValueError):
        validate_intervention(
            LensScoreRequest(prompt="x", directional_edits=[edit(**kwargs)]),
            layers=1,
            hidden=3,
            tokens=8,
        )


def test_valid_clamp_and_span_readout_bounds():
    validate_intervention(
        LensScoreRequest(
            prompt="x",
            directional_edits=[edit()],
            capture_layers=[0],
            capture_span_positions=[1, 2],
            full_readout=True,
        ),
        layers=1,
        hidden=3,
        tokens=8,
    )
    with pytest.raises(ValueError):
        validate_intervention(
            LensScoreRequest(
                prompt="x", capture_layers=[0], capture_span_positions=[8]
            ),
            layers=1,
            hidden=3,
            tokens=8,
        )


def test_homogeneous_batch_equals_independent_request_edits():
    from gleipnir.serving.lens_projection import apply_projection_batch

    h = torch.arange(27, dtype=torch.float32).reshape(9, 3) / 10
    independent = h.clone()
    batch = h.clone()
    d = torch.tensor([1.0, 0.0, 0.0])
    apply_projection(independent, h, d, 0.25, 1.0, 2.0, 0, 3, 0, 5)
    apply_projection(independent, h, d, 0.25, 1.0, 2.0, 3, 7, 2, 5)
    apply_projection_batch(batch, h, d, 0.25, 1.0, 2.0, 7, [6])
    torch.testing.assert_close(batch, independent)
    assert torch.equal(batch[7:], h[7:])


def test_span_mean_reduces_across_chunks_and_projection_cleans_up(monkeypatch):
    from types import SimpleNamespace

    from vllm_lens._helpers._serialize import deserialize_tensor

    import gleipnir.serving.lens_worker as module

    cfg = dict(
        request_id="a",
        capture_layers=[0],
        capture_positions="last",
        capture_span_positions=[1, 2, 5],
        steering_vectors=[],
        directional_edits=[edit(beta=0)],
        projection_identity="same",
    )
    worker = module.MonitorLensExtension()
    worker._lens_buffers = {}
    worker._lens_vectors = {}
    worker._lens_edits = {}
    worker._lens_layers = [0]
    state = SimpleNamespace(
        pooling_params=SimpleNamespace(extra_kwargs={"gleipnir_lens": cfg}),
        prompt_token_ids=[0] * 8,
    )
    batch = SimpleNamespace(
        req_ids=["a"], num_reqs=1, num_computed_tokens_cpu=torch.tensor([0])
    )
    worker.model_runner = SimpleNamespace(
        input_batch=batch,
        query_start_loc=SimpleNamespace(np=torch.tensor([0, 4])),
        requests={"a": state},
    )
    monkeypatch.setattr(module, "is_forward_context_available", lambda: True)
    first = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    second = torch.arange(12, 24, dtype=torch.float32).reshape(4, 3)
    worker._lens_forward(0, first)
    batch.num_computed_tokens_cpu = torch.tensor([4])
    worker._lens_forward(0, second)
    result = worker.lens_collect("a")
    actual = deserialize_tensor(result["activations"]["residual_span_mean"])
    torch.testing.assert_close(
        actual[0], torch.stack([first[1], first[2], second[1]]).mean(0)
    )
    assert result["span_capture_count"] == 3 and result["activation_positions"] == [7]
    assert not worker._lens_buffers and not worker._lens_edits
