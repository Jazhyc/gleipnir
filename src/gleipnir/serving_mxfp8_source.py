"""Source-isolated numerical variant of the pinned NVIDIA D256 MXFP8 kernel."""

import ast


def hardware_exp2_source(source: str) -> str:
    """Replace the two mixed polynomial helpers with hardware vector exp2."""
    tree = ast.parse(source)
    names = {"_exp2_chunk0a_mixed", "_exp2_chunk1b_mixed"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != 2 or {n.name for n in nodes} != names:
        raise ValueError("MXFP8 exponential helper source drift")
    lines = source.splitlines(keepends=True)
    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        if not node.args.args or node.args.args[0].arg != "vec":
            raise ValueError("MXFP8 exponential helper signature drift")
        lines[node.body[0].lineno - 1 : node.end_lineno] = [
            "    return cute.math.exp2(vec, fastmath=True)\n"
        ]
    return "".join(lines)
