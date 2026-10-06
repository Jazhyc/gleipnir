"""Exercise output alias/tail safety and preserved GDN preprocessing on CPU."""

import json

import pytest
import torch

from experiments.b200_attention_gdn_serving.run import EXPERIMENT, resolve_condition
from gleipnir.serving_gdn_direct_output import (
    destination_view,
    make_forward,
    validate_native,
    validate_runtime,
)


def inputs():
    q, k = [torch.randn(1, 5, 2, 128, dtype=torch.bfloat16) for _ in range(2)]
    v = torch.randn(1, 5, 4, 128, dtype=torch.bfloat16)
    g = torch.full((1, 5, 4), -0.2, dtype=torch.bfloat16)
    beta = torch.ones_like(g)
    state = torch.zeros(1, 4, 128, 128, dtype=torch.bfloat16)
    return q, k, v, g, beta, state, True, torch.tensor([0, 5])


def test_direct_output_aliases_only_active_view_and_preserves_preprocessing():
    args = inputs()
    calls, normalized = [], []

    def kernel(**kwargs):
        calls.append(kwargs)
        value = kwargs["output"]
        value.copy_(kwargs["v"])
        return value, kwargs["initial_state"]

    def normalize(x):
        normalized.append(x)
        return x * 2

    backing = torch.full((9, 4, 128), -123, dtype=torch.bfloat16)
    destination = backing[1:]
    output, state = make_forward(kernel, normalize)(*args, core_attn_out=destination)
    assert output.data_ptr() == destination.data_ptr()
    assert torch.equal(output, args[2])
    assert len(normalized) == 2
    assert torch.equal(calls[0]["q"], args[0].squeeze(0) * 2)
    assert torch.equal(calls[0]["g"], args[3].squeeze(0).float().exp())
    assert calls[0]["beta"].dtype == state.dtype == torch.float32
    assert torch.all(backing[:1] == -123) and torch.all(backing[6:] == -123)


def test_optional_normalization_and_final_state_match_wrapper_contract():
    args = inputs()

    def kernel(**kwargs):
        assert kwargs["output"] is None
        assert not kwargs["output_final_state"]
        assert torch.equal(kwargs["q"], args[0].squeeze(0))
        return kwargs["v"]

    output, state = make_forward(kernel, lambda _: pytest.fail("normalized"))(
        *args[:6], False, args[7], use_qk_l2norm_in_kernel=False
    )
    assert torch.equal(output, args[2]) and state is None


@pytest.mark.parametrize("kind", ["short", "strided", "dtype", "unaligned"])
def test_destination_layout_rejections(kind):
    q, _, v, *_ = inputs()
    destination = torch.empty(6 * 4 * 128, dtype=torch.bfloat16)
    if kind == "short":
        destination = destination[:10]
    elif kind == "strided":
        destination = torch.empty(10, 4, 128, dtype=torch.bfloat16)[::2]
    elif kind == "dtype":
        destination = destination.float()
    else:
        destination = destination[1:]
    with pytest.raises(ValueError, match="direct GDN output"):
        destination_view(destination, q, v)


def test_destination_aliasing_and_ignored_output_are_rejected():
    args = inputs()
    forward = make_forward(
        lambda **kw: (kw["output"].clone(), kw["initial_state"]), lambda x: x
    )
    with pytest.raises(ValueError, match="aliases an input"):
        forward(*args, core_attn_out=args[2])
    with pytest.raises(ValueError, match="did not return"):
        forward(*args, core_attn_out=torch.empty_like(args[2]))


def test_native_and_live_admission_fail_closed():
    native = {
        "passed": True,
        "intervention": "flashinfer_gdn_direct_output",
        "checks": [
            {"lengths": [n], "passed": True}
            for n in (1, 17, 129, 641, 4096, 16384, 32768)
        ],
        "graph_replay_passed": True,
        "isolation_passed": True,
        "zero_passed": True,
        "sources": {"source": "hash"},
    }
    validate_native(native)
    with pytest.raises(ValueError, match="native admission"):
        validate_native({**native, "graph_replay_passed": False})
    live = {
        "passed": True,
        "worker_pid": 10,
        "validation_path": "receipt",
        "calls": [{"layer": i, "direct": True} for i in range(24)],
    }
    validate_runtime(live, 10, "receipt")
    for bad in ({**live, "worker_pid": 11}, {**live, "calls": live["calls"][:-1]}):
        with pytest.raises(ValueError, match="live dispatch"):
            validate_runtime(bad, 10, "receipt")


def test_direct_condition_requires_bound_flashinfer_worker():
    condition = json.loads((EXPERIMENT / "gdn_direct_output.json").read_text())
    resolve_condition(condition, {})
    for bad in (
        {**condition, "gdn_direct_output_validation": ""},
        {**condition, "worker_cls": condition["worker_cls"].replace("DirectGdn", "")},
    ):
        with pytest.raises(ValueError, match="direct GDN output"):
            resolve_condition(bad, {})
