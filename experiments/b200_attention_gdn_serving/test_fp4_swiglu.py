"""Check the frozen-weight permutation's gate pairing and tile-scale lineage."""

import pytest

from gleipnir.serving_fp4_swiglu import adapt_source, interleaved_rows


@pytest.mark.parametrize("n", [64, 128, 18432])
def test_interleaving_preserves_gate_pair_and_scale_groups(n):
    rows = interleaved_rows(n)
    assert sorted(rows) == list(range(n))
    for block in range(n // 64):
        up, gate = rows[block * 64 : block * 64 + 32], rows[
            block * 64 + 32 : (block + 1) * 64
        ]
        assert all(u == g + n // 2 for u, g in zip(up, gate, strict=True))
    # A packed weight's 16-row scale tile remains intact after permutation.
    for start in range(0, n, 16):
        group = rows[start : start + 16]
        assert group[0] % 16 == 0
        assert group == list(range(group[0], group[0] + 16))


@pytest.mark.parametrize("n", [0, -64, 32, 65])
def test_invalid_gate_width_rejected(n):
    with pytest.raises(ValueError, match="divisible by 64"):
        interleaved_rows(n)


def test_template_drift_rejected_before_codegen():
    with pytest.raises(ValueError, match="source drift"):
        adapt_source("unknown upstream kernel")
