"""Independent CPU decoding for packed E2M1 values and NVFP4 block scales."""

from __future__ import annotations

import numpy as np

E2M1_VALUES = np.asarray(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)


def decode_nvfp4(
    packed: np.ndarray, block_scales: np.ndarray, global_scale: float
) -> np.ndarray:
    """Decode low-nibble-first row-major FP4 with one E4M3 scale per 16 values."""
    if packed.ndim != 2 or packed.dtype != np.uint8:
        raise ValueError("Packed NVFP4 must be a two-dimensional uint8 array")
    rows, half_width = packed.shape
    width = half_width * 2
    if width % 16 or block_scales.shape != (rows, width // 16):
        raise ValueError("NVFP4 block-scale coverage mismatch")
    if not np.isfinite(global_scale) or global_scale <= 0:
        raise ValueError("NVFP4 global scale must be finite and positive")
    if not np.isfinite(block_scales).all() or (block_scales < 0).any():
        raise ValueError("NVFP4 block scales must be finite and nonnegative")
    codes = np.empty((rows, width), dtype=np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    return (
        E2M1_VALUES[codes]
        * np.repeat(block_scales.astype(np.float32), 16, axis=1)
        * np.float32(global_scale)
    )
