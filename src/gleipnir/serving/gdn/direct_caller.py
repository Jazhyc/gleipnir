"""Generate a narrowly guarded caller adaptation from pinned vLLM source."""

import ast
import textwrap


def build_core_source(source: str) -> str:
    """Forward the ordinary-prefill buffer and omit its now-redundant copy."""
    tree = ast.parse(textwrap.dedent(source))
    calls = 0
    copies = 0

    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal calls
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
                and node.func.attr == "chunk_gated_delta_rule"
            ):
                if any(k.arg == "core_attn_out" for k in node.keywords):
                    raise ValueError("vLLM caller already forwards a GDN destination")
                node.keywords.append(
                    ast.keyword(
                        arg="core_attn_out",
                        value=ast.parse(
                            "core_attn_out[:num_actual_tokens] if "
                            "spec_sequence_masks is None and not split_non_spec "
                            "else None",
                            mode="eval",
                        ).body,
                    )
                )
                calls += 1
            return node

        def visit_Assign(self, node):
            nonlocal copies
            self.generic_visit(node)
            if (
                len(node.targets) == 1
                and ast.unparse(node.targets[0]) == "core_attn_out[:num_actual_tokens]"
                and ast.unparse(node.value) == "core_attn_out_non_spec.squeeze(0)"
            ):
                copies += 1
                return ast.copy_location(
                    ast.If(
                        test=ast.parse(
                            "attn_metadata.num_prefills == 0 or "
                            "spec_sequence_masks is not None or split_non_spec",
                            mode="eval",
                        ).body,
                        body=[node],
                        orelse=[],
                    ),
                    node,
                )
            return node

    tree = Rewrite().visit(tree)
    if calls != 1 or copies != 1:
        raise ValueError(f"unsupported vLLM GDN caller: calls={calls}, copies={copies}")
    ast.fix_missing_locations(tree)
    return (
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"
        "# Derived from the source-bound installed vLLM GDN caller.\n"
        + ast.unparse(tree)
        + "\n"
    )
