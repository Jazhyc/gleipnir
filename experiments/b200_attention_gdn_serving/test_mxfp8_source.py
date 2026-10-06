"""Retain native helper signatures/decorators and reject source drift."""

import ast

import pytest

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
