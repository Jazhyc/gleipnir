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


def test_cute_tuning_bounds_candidates_without_inventing_tactics(monkeypatch):
    import sys
    from types import ModuleType

    from gleipnir.serving_fp4_backends import bounded_cute_tuning

    tactics = [
        (tile, cluster, swap, prefetch)
        for tile in ((128, 256), (256, 128), (128, 128), (256, 256))
        for cluster in ((1, 1), (2, 1), (4, 1))
        for swap in (False, True)
        for prefetch in (False, True)
    ]
    base = ModuleType("flashinfer.gemm.gemm_base")
    base._get_sm100_block_scaled_tactics = lambda *args: tactics
    gemm = ModuleType("flashinfer.gemm")
    gemm.gemm_base = base
    parent = ModuleType("flashinfer")
    parent.gemm = gemm
    monkeypatch.setitem(sys.modules, "flashinfer", parent)
    monkeypatch.setitem(sys.modules, "flashinfer.gemm", gemm)
    audit = bounded_cute_tuning()
    chosen = base._get_sm100_block_scaled_tactics(32768)
    assert len(chosen) == len(set(chosen)) == 8
    assert set(chosen) <= set(tactics)
    assert {t[2] for t in chosen} == {False, True}
    assert all(t[1] == (2, 1) and t[3] for t in chosen)
    assert audit[0]["valid_count"] == len(tactics)


def test_cute_aliases_preserve_target_and_existing_functions():
    from types import SimpleNamespace

    from gleipnir.serving_fp4_cute_compat import install_aliases

    tensor_factory = object()
    existing_like = object()
    cute = SimpleNamespace(
        make_rmem_tensor=tensor_factory, make_fragment_like=existing_like
    )
    assert install_aliases(cute) == {"make_fragment": "make_rmem_tensor"}
    assert cute.make_fragment is tensor_factory
    assert cute.make_fragment_like is existing_like
    assert install_aliases(cute) == {}
    with pytest.raises(ValueError, match="missing CuTe compatibility target"):
        install_aliases(SimpleNamespace())
