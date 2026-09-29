"""Independent FP4 tie behavior and fail-closed experimental packer inputs."""

import numpy as np
import pytest
import torch

from gleipnir.nvfp4_pack import pack_nvfp4
from gleipnir.nvfp4_reference import E2M1_VALUES, encode_e2m1


def test_e2m1_codes_and_negative_zero_roundtrip():
    np.testing.assert_array_equal(encode_e2m1(E2M1_VALUES), np.arange(16))


def test_midpoints_round_to_even():
    midpoints = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], np.float32)
    expected = np.array([0, 2, 2, 4, 4, 6, 6], np.uint8)
    np.testing.assert_array_equal(encode_e2m1(midpoints), expected)
    np.testing.assert_array_equal(encode_e2m1(-midpoints), expected | 8)
    np.testing.assert_array_equal(encode_e2m1(midpoints + 0.001), np.arange(1, 8))
    np.testing.assert_array_equal(encode_e2m1(midpoints - 0.001), np.arange(7))


def test_e2m1_saturation_and_nonfinite_rejection():
    np.testing.assert_array_equal(encode_e2m1(np.array([100, -100])), [7, 15])
    with pytest.raises(ValueError, match="finite"):
        encode_e2m1(np.array([np.nan]))


def test_triton_packer_rejects_cpu_emulation():
    with pytest.raises(ValueError, match="CUDA"):
        pack_nvfp4(torch.zeros(32, 64, dtype=torch.bfloat16), torch.ones(1))
