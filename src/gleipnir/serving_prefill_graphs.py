"""Bound large-prefill graph padding without changing FP4 producer bands."""

import ast
import textwrap


def adapt_profile_cache_source(source: str) -> str:
    """Size the temporary piecewise profiling cache by requests, not tokens."""
    tree = ast.parse(textwrap.dedent(source))
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "min_blocks"
    ]
    if len(candidates) != 1 or ast.unparse(candidates[0].value) != (
        "self.compilation_config.max_cudagraph_capture_size or 1"
    ):
        raise ValueError("profiling cache allocator source changed")
    candidates[0].value = ast.parse(
        "self.scheduler_config.max_num_seqs", mode="eval"
    ).body
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def producer_band(rows: int) -> str:
    if rows < 1536:
        return "frost"
    return "whole_row" if rows <= 4096 else "native_block"


def graph_padding_allowed(rows: int, padded: int, limit: float) -> bool:
    """Keep actual and padded rows in the same accepted arithmetic branch."""
    return (
        rows > 0
        and padded >= rows
        and padded <= rows * (1 + limit)
        and producer_band(rows) == producer_band(padded)
    )


def validate_graph_config(condition: dict) -> dict:
    """Allow only piecewise capture and bounded, explicitly enumerated shapes."""
    config = condition["compilation_config"]
    policy = condition["prefill_graphs"]
    sizes = config.get("cudagraph_capture_sizes", [])
    if (
        set(config) != {"cudagraph_mode", "cudagraph_capture_sizes"}
        or config["cudagraph_mode"] != "PIECEWISE"
        or not sizes
        or len(sizes) > 20
        or any(type(s) is not int or s <= 0 or s > 32768 for s in sizes)
        or sizes != sorted(set(sizes))
        or sizes[-1] != 32768
        or set(policy)
        != {"max_padding_fraction", "dispatcher_sha256", "model_runner_sha256"}
        or not 0 <= policy["max_padding_fraction"] <= 0.125
        or len(policy["dispatcher_sha256"]) != 64
        or len(policy["model_runner_sha256"]) != 64
    ):
        raise ValueError("unsupported prefill graph configuration")
    return config
