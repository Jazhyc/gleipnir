"""Packed-value order, signs and scale coverage have independent references."""

import numpy as np
import pytest

from gleipnir.nvfp4_reference import decode_nvfp4


def test_decode_nibble_order_signs_and_two_distinct_blocks():
    packed = np.asarray(
        [[0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x0F] * 2], dtype=np.uint8
    )
    decoded = decode_nvfp4(packed, np.asarray([[2, 3]], dtype=np.float32), 0.5)
    expected = np.asarray(
        [0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6, 0],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(decoded[0, :16], expected)
    np.testing.assert_array_equal(decoded[0, 16:], expected * 1.5)
    assert np.signbit(decoded[0, 7])


def test_decode_rejects_corrupt_scale_coverage_and_nonfinite_scales():
    packed = np.zeros((2, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="coverage"):
        decode_nvfp4(packed, np.ones((2, 2)), 1)
    with pytest.raises(ValueError, match="finite"):
        decode_nvfp4(packed, np.asarray([[1], [np.nan]]), 1)
    with pytest.raises(ValueError, match="positive"):
        decode_nvfp4(packed, np.ones((2, 1)), 0)
