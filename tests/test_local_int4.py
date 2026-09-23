"""Packed signed-nibble layout used by the independent GEMM reference."""

import torch

from experiments.local_inference.int4_bench import unpack


def test_unpack_signed_nibbles_and_row_boundaries():
    packed = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76], [0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8
    )
    expected = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [-8, -7, -6, -5, -4, -3, -2, -1]]
    )
    assert torch.equal(unpack(packed), expected)


def test_unpack_zero():
    assert torch.equal(
        unpack(torch.zeros((2, 8), dtype=torch.uint8)), torch.zeros(2, 16)
    )
