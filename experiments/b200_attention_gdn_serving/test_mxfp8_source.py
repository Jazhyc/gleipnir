"""Retain native helper signatures/decorators and reject source drift."""

import ast

import pytest

from gleipnir.serving.mxfp8_compile_compat import typed_kv_coordinates_source
from gleipnir.serving_mxfp8_source import hardware_exp2_source


def test_hardware_exp2_preserves_helpers_and_surrounding_kernel_source():
    source = """KERNEL_CONSTANT = 127
@decorator
def _exp2_chunk0a_mixed(vec, apply_mask, dense=False, softmax_half=0):
    values = []
    return polynomial(vec)
@decorator
def _exp2_chunk1b_mixed(vec, dense=False, softmax_half=0):
    return approximation(vec)
def kernel():
    return KERNEL_CONSTANT
"""
    transformed = hardware_exp2_source(source)
    tree = ast.parse(transformed)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert [n.name for n in functions] == [
        "_exp2_chunk0a_mixed",
        "_exp2_chunk1b_mixed",
        "kernel",
    ]
    assert all(len(n.decorator_list) == 1 for n in functions[:2])
    assert transformed.count("return cute.math.exp2(vec, fastmath=True)") == 2
    assert "KERNEL_CONSTANT = 127" in transformed
    assert "return KERNEL_CONSTANT" in transformed
    with pytest.raises(ValueError, match="source drift"):
        hardware_exp2_source(source.replace("_exp2_chunk1b_mixed", "renamed_helper"))
    with pytest.raises(ValueError, match="signature drift"):
        hardware_exp2_source(source.replace("(vec,", "(other,"))


def test_typed_coordinates_only_initialize_conditionally_assigned_temporaries():
    source = """@cute.jit
def _tmaldg_warp_group(flag, count):
    state = start()
    if flag:
        k_row, k_page, k_sf_tile, v_row, v_page, v_sf_group = _kv_tile_coords(0)
        use(k_row, v_row)
        if count > 1:
            k_row, k_page, k_sf_tile, v_row, v_page, v_sf_group = _kv_tile_coords(1)
            use(k_row, v_row)
    for i in cutlass.range(2, count):
        k_row, k_page, k_sf_tile, v_row, v_page, v_sf_group = _kv_tile_coords(i)
        use(k_row, v_row)
"""
    transformed = typed_kv_coordinates_source(source)
    tree = ast.parse(transformed)
    original = ast.parse(source)
    assert ast.dump(tree.body[0].body[6:][0]) == ast.dump(original.body[0].body[0])
    tree.body[0].body = tree.body[0].body[6:]
    assert ast.dump(tree) == ast.dump(original)
    assert transformed.count(" = cutlass.Int32(0)") == 6
    with pytest.raises(ValueError, match="already initialized"):
        typed_kv_coordinates_source(transformed)
    with pytest.raises(ValueError, match="assignment drift"):
        typed_kv_coordinates_source(source.replace("_kv_tile_coords(1)", "other(1)"))
    with pytest.raises(ValueError, match="producer source drift"):
        typed_kv_coordinates_source(source.replace("_tmaldg_warp_group", "other"))
