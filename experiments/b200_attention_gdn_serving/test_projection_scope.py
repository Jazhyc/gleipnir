"""Protect non-target gates, full-attention and visual projections."""

import pytest

from gleipnir.serving_precision import is_gdn_projection


@pytest.mark.parametrize("projection", ["in_proj_qkvz", "out_proj"])
def test_only_large_decoder_gdn_projections_are_selected(projection):
    assert is_gdn_projection(f"language_model.model.layers.0.linear_attn.{projection}")
    assert is_gdn_projection(f"model.layers.30.linear_attn.{projection}")
    assert not is_gdn_projection(f"visual.layers.0.linear_attn.{projection}")


@pytest.mark.parametrize(
    "prefix",
    [
        "model.layers.0.linear_attn.in_proj_ba",
        "model.layers.3.self_attn.o_proj",
        "model.layers.3.mlp.down_proj",
        "linear_attn.out_proj",
        "model.layers.0.linear_attn.out_proj_extra",
    ],
)
def test_other_projections_are_not_quantized(prefix):
    assert not is_gdn_projection(prefix)
