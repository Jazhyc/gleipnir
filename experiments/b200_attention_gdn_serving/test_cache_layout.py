"""Audit native NVFP4 K/V pairs as well as ordinary combined cache tensors."""

from types import SimpleNamespace

import pytest

from gleipnir.serving_precision import describe_attention_cache


def test_separate_native_kv_views_preserve_shapes_and_precision():
    k = SimpleNamespace(dtype="torch.uint8", shape=(7, 4, 32, 128))
    v = SimpleNamespace(dtype="torch.uint8", shape=(7, 4, 32, 128))
    assert describe_attention_cache((k, v)) == {
        "dtype": "torch.uint8",
        "shape": [[7, 4, 32, 128]] * 2,
        "layout": "kv_pair",
    }
    assert describe_attention_cache(k)["layout"] == "tensor"


def test_malformed_or_mixed_precision_kv_pairs_fail_closed():
    k = SimpleNamespace(dtype="torch.uint8", shape=(7,))
    v = SimpleNamespace(dtype="torch.bfloat16", shape=(7,))
    for pair in [(k,), (k, k, k), (k, v)]:
        with pytest.raises(ValueError, match="invalid native K/V"):
            describe_attention_cache(pair)
