"""Scoped native grouped-head GDN calls, preserving upstream precision/hooks."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import MethodType
from typing import Any

import torch


def grouped_gdn_forward(original: Callable) -> tuple[Callable, dict]:
    """Remove only the verified upstream Q/K replication block."""
    hook_child = None
    closure = inspect.getclosurevars(original).nonlocals
    if {"child_module_name", "forward_func"} <= closure.keys():
        hook_child = closure["child_module_name"]
        original = closure["forward_func"]
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise ValueError("expected a GDN forward function")
    if function.decorator_list:
        if hook_child is None or [ast.unparse(d) for d in function.decorator_list] != [
            f"force_accelerate_hooks({hook_child!r})"
        ]:
            raise ValueError("unsupported GDN forward decorator")
        function.decorator_list = []
    expected = ast.parse(
        "if self.num_v_heads // self.num_k_heads > 1:\n"
        "    query = query.repeat_interleave("
        "self.num_v_heads // self.num_k_heads, dim=2)\n"
        "    key = key.repeat_interleave("
        "self.num_v_heads // self.num_k_heads, dim=2)\n"
    ).body[0]
    matches = [s for s in function.body if ast.dump(s) == ast.dump(expected)]
    if len(matches) != 1:
        raise ValueError("upstream GDN head replication block changed")
    function.body.remove(matches[0])
    function.body[:0] = ast.parse(
        "if cache_params is not None:\n"
        "    raise ValueError('grouped GDN screen requires cache-free training')\n"
    ).body
    function.name = "gleipnir_grouped_gdn_forward"
    ast.fix_missing_locations(tree)
    transformed = ast.unparse(tree) + "\n"
    scope = dict(original.__globals__)
    exec(
        compile("from __future__ import annotations\n" + transformed, __file__, "exec"),
        scope,
    )
    forward = scope[function.name]
    if hook_child is not None:
        forward = original.__globals__["force_accelerate_hooks"](hook_child)(forward)
    return forward, {
        "upstream_forward_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "transformed_forward_sha256": hashlib.sha256(transformed.encode()).hexdigest(),
        "accelerate_hook_child": hook_child,
        "removed_repeat_interleave_calls": 2,
    }


@contextmanager
def grouped_gdn_context(model: torch.nn.Module) -> Iterator[dict[str, Any]]:
    """Retain shared Q/K heads for a caller already using native grouped FlashQLA."""
    modules = [
        (name, m)
        for name, m in model.named_modules()
        if type(m).__name__ == "Qwen3_5GatedDeltaNet"
    ]
    if len(modules) != 24:
        raise ValueError("grouped GDN screen requires all 24 Qwen4B GDN layers")
    replacements, transforms = [], []
    for name, module in modules:
        if (
            module.num_v_heads <= module.num_k_heads
            or module.num_v_heads % module.num_k_heads
        ):
            raise ValueError("expected divisible grouped Q/K and value heads")
        original = inspect.unwrap(module.forward)
        original = getattr(original, "__func__", original)
        if original is not type(module).forward:
            raise ValueError("unsupported GDN forward override")
        function, metadata = grouped_gdn_forward(original)
        bound = MethodType(function, module)
        if getattr(module.forward, "_torchdynamo_disable", False):
            bound = torch.compiler.disable(bound)
        replacements.append((module, bound))
        transforms.append({"module": name, **metadata})
    missing = object()
    saved = [(m, m.__dict__.get("forward", missing)) for m, _ in replacements]
    try:
        for module, function in replacements:
            module.forward = function
        yield {
            "modules": [name for name, _ in modules],
            "query_key_heads": modules[0][1].num_k_heads,
            "value_heads": modules[0][1].num_v_heads,
            "precision_boundary": "unchanged_bf16_qkv_fp32_gates_norm",
            "parameter_identity_preserved": True,
            "source_transforms": transforms,
        }
    finally:
        for module, original in saved:
            if original is missing:
                del module.forward
            else:
                module.forward = original
