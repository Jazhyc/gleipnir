"""Native backward wrapper rejects layout drift and restores scoped patches."""

import pytest
import torch

from gleipnir.training.mxfp8_backward import native_attention


def operands(heads=16):
    return (
        torch.zeros(3, heads, 256),
        torch.zeros(3, 4, 256),
        torch.zeros(3, 4, 256),
        torch.tensor([0, 3], dtype=torch.int32),
    )


def test_native_backward_rejects_changed_head_contract():
    q, k, v, cuts = operands(8)
    with pytest.raises(ValueError, match="heads"):
        native_attention(q, k, v, cuts, 3)


def test_native_backward_rejects_unsupported_context():
    with pytest.raises(ValueError, match="context"):
        native_attention(*operands(), 32769)


def test_native_backward_restores_patches_after_failure(monkeypatch):
    import gleipnir.nvidia_mxfp8_varlen_attention as packed
    import gleipnir.nvidia_mxfp8_varlen_repack as repack

    old_plan, old_singleton = packed.forward_plan, repack.singleton_forward

    def failure(*args, **kwargs):
        assert packed.forward_plan is not old_plan
        assert repack.singleton_forward is not old_singleton
        raise RuntimeError("compile failure")

    monkeypatch.setattr(packed, "packed_attention", failure)
    with pytest.raises(RuntimeError, match="compile failure"):
        native_attention(*operands(), 3)
    assert packed.forward_plan is old_plan
    assert repack.singleton_forward is old_singleton
