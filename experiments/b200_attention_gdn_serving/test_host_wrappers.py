"""Bindings must preserve role order and cache storage-aware read-only views."""

from types import SimpleNamespace

import pytest

from experiments.b200_attention_gdn_serving.host_wrapper_compare import select_direct
from gleipnir.serving_frost_wrappers import WeightViews, role_indices


def test_role_mapping_rejects_extra_or_missing_operands():
    roles = [object() for _ in range(6)]
    plan = SimpleNamespace(
        **dict(
            zip(
                [
                    "a_tensor",
                    "b_tensor",
                    "sfa_tensor",
                    "sfb_tensor",
                    "scale_tensor",
                    "output_tensor",
                ],
                roles,
                strict=True,
            )
        )
    )
    plan.jit = SimpleNamespace(bound=roles[::-1], lowered=lambda *args: None)
    plan.workspace_bytes = 0
    assert role_indices(plan) == (5, 4, 3, 2, 1, 0)
    plan.jit.bound = [*roles, object()]
    with pytest.raises(ValueError, match="six-tensor"):
        role_indices(plan)
    plan.jit.bound = roles
    plan.workspace_bytes = 1
    with pytest.raises(ValueError, match="zero-workspace"):
        role_indices(plan)


class Tensor:
    shape = (2, 4)
    dtype = "packed"
    device = "cuda:0"

    def __init__(self, pointer):
        self.pointer = pointer
        self.strides = (4, 1)

    def data_ptr(self):
        return self.pointer

    def stride(self):
        return self.strides


def test_cached_views_invalidate_on_storage_or_layout_and_stay_bounded():
    a, b = Tensor(16), Tensor(32)
    count = []

    def factory(*args):
        count.append(args)
        return tuple(object() for _ in args)

    cache = WeightViews(limit=2)
    first = cache.get(a, b, factory)
    assert cache.get(a, b, factory) is first and len(count) == 1
    a.pointer = 64
    second = cache.get(a, b, factory)
    assert second is not first and len(count) == 2
    b.strides = (1, 2)
    assert cache.get(a, b, factory) is not second
    assert len(cache.entries) == 2
    assert cache.get(a, b, factory) is not second and len(count) == 4


def test_selection_rejects_inconsistent_speed_or_score_regression():
    comparison = {
        "c1": {
            "latency_p50": {"original": 1.0, "direct": 0.99},
            "latency_p95": {"original": 1.0, "direct": 1.0},
            "paired_scores": {"score": {"max_absolute_difference": 0}},
        },
        "c128": {
            "paired_throughput": {"median": 1.02, "ratios": [1.02] * 6},
            "ranking": {"auroc_delta": {"macro": 0, "pooled": 0}},
        },
    }
    assert select_direct(comparison)
    comparison["c128"]["paired_throughput"]["ratios"] = [0.99] * 2 + [1.02] * 4
    assert not select_direct(comparison)
    comparison["c128"]["paired_throughput"]["ratios"] = [1.02] * 6
    comparison["c1"]["paired_scores"]["score"]["max_absolute_difference"] = 0.001
    assert not select_direct(comparison)
