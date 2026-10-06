"""Check scale-layout permutations and reject malformed backend requests."""

import pytest

from gleipnir.serving_fp4_backends import linear_scale_offsets


def test_scale_offsets_form_complete_swizzle_atoms():
    offsets = linear_scale_offsets(128, 64)
    assert sorted(v for row in offsets for v in row) == list(range(512))
    assert offsets[0] == [0, 1, 2, 3]
    assert offsets[32] == [4, 5, 6, 7]
    assert offsets[1] == [16, 17, 18, 19]
    extended = linear_scale_offsets(256, 128)
    assert sorted(v for row in extended for v in row) == list(range(2048))


@pytest.mark.parametrize("rows,k", [(0, 64), (-1, 64), (128, 63), (128, 0)])
def test_scale_offsets_reject_unsupported_geometry(rows, k):
    with pytest.raises(ValueError, match="positive rows"):
        linear_scale_offsets(rows, k)
