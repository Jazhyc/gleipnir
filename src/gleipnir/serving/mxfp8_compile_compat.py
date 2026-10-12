"""Opt-in source repair for typed TMA coordinates in the pinned MXFP8 kernel."""

import ast


def typed_kv_coordinates_source(source: str) -> str:
    """Give conditionally assigned TMA coordinates stable loop-carried types."""
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_tmaldg_warp_group"
    ]
    if len(functions) != 1:
        raise ValueError("MXFP8 TMA producer source drift")
    function = functions[0]
    names = ("k_row", "k_page", "k_sf_tile", "v_row", "v_page", "v_sf_group")
    assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Tuple)
        and tuple(getattr(item, "id", None) for item in node.targets[0].elts) == names
    ]
    if len(assignments) != 3 or any(
        not isinstance(node.value, ast.Call)
        or not isinstance(node.value.func, ast.Name)
        or node.value.func.id != "_kv_tile_coords"
        for node in assignments
    ):
        raise ValueError("MXFP8 KV coordinate assignment drift")
    if any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        )
        for node in ast.walk(function)
    ):
        raise ValueError("MXFP8 KV coordinates already initialized or source drift")
    first = function.body[0]
    lines = source.splitlines(keepends=True)
    indent = lines[first.lineno - 1][: first.col_offset]
    initialization = [indent + name + " = cutlass.Int32(0)\n" for name in names]
    lines[first.lineno - 1 : first.lineno - 1] = initialization
    return "".join(lines)
