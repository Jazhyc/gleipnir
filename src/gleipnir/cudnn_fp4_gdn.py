"""Scoped frozen GDN projections using the validated NVIDIA NVFP4 primitives."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from types import MethodType
from typing import Any

import torch

from gleipnir.cudnn_fp4_mlp import fp4_epilogue_linear


def projection_forward(self: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Keep the frozen BF16 parameter; return BF16 from an FP4 contraction."""
    return fp4_epilogue_linear(x, self.weight, None)


@lru_cache(maxsize=8)
def merged_gdn_forward(original: Callable) -> tuple[Callable, dict[str, str]]:
    """Replace only the upstream QKV/Z assignments; retain its recurrent shell."""
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
        decorators = [ast.unparse(d) for d in function.decorator_list]
        if hook_child is None or decorators != [
            f"force_accelerate_hooks({hook_child!r})"
        ]:
            raise ValueError("unsupported GDN forward decorator")
        function.decorator_list = []
    expected = {
        "mixed_qkv = self.in_proj_qkv(hidden_states)": "qkv",
        "z = self.in_proj_z(hidden_states)": "z",
    }
    found = []
    body = []
    for statement in function.body:
        kind = expected.get(ast.unparse(statement))
        if kind == "qkv":
            found.append(kind)
            body.extend(
                ast.parse(
                    "mixed_qkv, z = _gleipnir_fp4_linear("
                    "hidden_states, self.in_proj_qkv.weight, self.in_proj_z.weight"
                    ").split((self.in_proj_qkv.out_features, "
                    "self.in_proj_z.out_features), dim=-1)"
                ).body
            )
        elif kind == "z":
            found.append(kind)
        else:
            body.append(statement)
    if found != ["qkv", "z"]:
        raise ValueError("upstream GDN projection assignments changed")
    function.body = body
    function.name = "gleipnir_merged_gdn_forward"
    ast.fix_missing_locations(tree)
    transformed = ast.unparse(tree) + "\n"
    scope = {**original.__globals__, "_gleipnir_fp4_linear": fp4_epilogue_linear}
    # Future annotations avoid executing upstream typing imports again.
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
    }


@contextmanager
def gdn_fp4_context(
    model: torch.nn.Module, *, merged_inputs: bool = True
) -> Iterator[dict[str, Any]]:
    """Change large projections without changing gates, recurrence or parameters."""
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if module.__class__.__name__ == "Qwen3_5GatedDeltaNet"
    ]
    if len(modules) != 24:
        raise ValueError("GDN FP4 screen requires all 24 Qwen3.5 GDN layers")
    for _, module in modules:
        for name in ("in_proj_qkv", "in_proj_z", "out_proj"):
            projection = getattr(module, name)
            weight = projection.weight
            if (
                type(projection) is not torch.nn.Linear
                or projection.bias is not None
                or weight.requires_grad
                or weight.dtype != torch.bfloat16
                or any(d % 64 for d in weight.shape)
            ):
                raise ValueError("GDN FP4 requires frozen biasless BF16 linear weights")
    receipts, replacements = [], []
    for name, module in modules:
        if merged_inputs:
            original = inspect.unwrap(module.forward)
            original = getattr(original, "__func__", original)
            if original is not type(module).forward:
                raise ValueError("unsupported GDN forward override")
            forward, receipt = merged_gdn_forward(original)
            replacements.append((module, forward))
            receipts.append({"module": name, **receipt})
        else:
            replacements.extend(
                (getattr(module, n), projection_forward)
                for n in ("in_proj_qkv", "in_proj_z")
            )
        replacements.append((module.out_proj, projection_forward))
    missing = object()
    saved = [
        (module, module.__dict__.get("forward", missing)) for module, _ in replacements
    ]
    try:
        for module, forward in replacements:
            bound = MethodType(forward, module)
            if getattr(module.forward, "_torchdynamo_disable", False):
                bound = torch.compiler.disable(bound)
            module.forward = bound
        yield {
            "modules": [name for name, _ in modules],
            "large_projections": ["in_proj_qkv", "in_proj_z", "out_proj"],
            "merged_qkv_z": merged_inputs,
            "shared_input_packing": merged_inputs,
            "base_forward_and_input_gradient": "nvfp4_hardware_pack_fused_descale",
            "small_gate_projections": "unchanged_bf16",
            "recurrence_and_precision_boundary": "unchanged",
            "original_parameters_preserved": True,
            "source_transforms": receipts,
        }
    finally:
        for module, prior in saved:
            if prior is missing:
                del module.forward
            else:
                module.forward = prior
