"""Normalize Goodfire nodes, then reuse the natural-cheats action renderer."""

from __future__ import annotations

import json
import re
from typing import Any

from experiments.adversarial_natural_cheats.trajectories import render


class RenderError(ValueError):
    """Unsupported source data must be counted, never silently normalized away."""


def calls(message: dict) -> list[dict]:
    output = []
    for call in message.get("tool_calls") or []:
        name = call["name"]
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as e:
                raise RenderError("invalid tool arguments") from e
        if not isinstance(args, dict) or not isinstance(name, str):
            raise RenderError("expected named tool call with JSON object arguments")
        output.append({"function": name, "arguments": args, "id": call.get("id")})
    return output


def visible_calls(message: dict, reasoning_tools: set[str]) -> list[dict]:
    return [c for c in calls(message) if c["function"] not in reasoning_tools]


def normalize(nodes: list[dict], reasoning_tools: set[str]) -> list[dict]:
    output = []
    hidden_ids = set()
    for i, node in enumerate(nodes):
        if node.get("parent") != (i - 1 if i else None):
            # Harness rewrites keep the sampled turn and its effective sibling.
            # Compare decoded calls; raw JSON may differ only in escaping.
            previous = nodes[i - 1] if i else {}
            sibling = (
                i > 0
                and node.get("parent") == previous.get("parent")
                and previous.get("sampled") is True
                and node.get("sampled") is False
                and previous["message"]["role"] == "assistant"
                and node["message"]["role"] == "assistant"
            )

            def signature(m: dict) -> list:
                return [(c["function"], c["arguments"]) for c in calls(m)]

            if sibling and signature(previous["message"]) == signature(node["message"]):
                hidden_ids.update(
                    c["id"]
                    for c in calls(node["message"])
                    if c["function"] in reasoning_tools and c["id"]
                )
                continue
            raise RenderError("nonlinear history beyond equivalent adjacent rewrite")
        m = node["message"]
        role = m["role"]
        text = m.get("content") or ""
        if not isinstance(text, str):
            raise RenderError("non-text message content")
        if role == "assistant":
            all_calls = calls(m)
            hidden_ids.update(
                c["id"]
                for c in all_calls
                if c["function"] in reasoning_tools and c["id"]
            )
            output.append(
                {
                    "role": role,
                    "text": "",
                    "tool_calls": [
                        c for c in all_calls if c["function"] not in reasoning_tools
                    ],
                }
            )
        elif role == "tool":
            if m.get("tool_call_id") in hidden_ids:
                continue
            output.append({"role": role, "text": text, "error": m.get("error")})
        elif role == "user":
            output.append({"role": role, "text": text})
        elif role != "system":
            raise RenderError(f"unknown role {role}")
    return output


def trajectory(nodes: list[dict], reasoning_tools: set[str]) -> str:
    return render(normalize(nodes, reasoning_tools))


def strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    return []


def quote_in_actions(flag: dict, nodes: list[dict], reasoning_tools: set[str]) -> bool:
    """Require the exact addressed quote in a retained call, not a think call.

    Accept raw and compact JSON plus decoded string arguments, because the judge
    and generic monitor use different JSON spacing and quote escaping. No search
    across nodes and no reattachment from reasoning or assistant prose.
    """
    index = flag.get("node_index")
    if type(index) is not int or not 0 <= index < len(nodes):
        raise RenderError("flag node_index outside rollout")
    m = nodes[index]["message"]
    if m["role"] != "assistant":
        raise RenderError("flag does not address an assistant node")
    if flag.get("channel") not in {"tool_call", "tool_calls"}:
        return False
    if (
        flag.get("quote_verified") is not True
        or flag.get("quote_found_in_release") is not True
    ):
        return False
    quote = flag.get("quote") or ""
    if not quote:
        return False

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    addressed_calls = visible_calls(m, reasoning_tools)
    if flag.get("channel") == "tool_call":
        address = re.search(r":tool-call-(\d+)$", flag.get("passage_id", ""))
        if not address:
            raise RenderError("generic tool_call lacks original call ordinal")
        ordinal = int(address.group(1)) - 1
        original = calls(m)
        if not 0 <= ordinal < len(original):
            raise RenderError("generic call ordinal outside source node")
        call = original[ordinal]
        addressed_calls = [] if call["function"] in reasoning_tools else [call]
    for call in addressed_calls:
        args = call["arguments"]
        candidates = strings(args) + [
            json.dumps(args, ensure_ascii=False),
            json.dumps(args, ensure_ascii=False, separators=(",", ":")),
        ]
        candidates += [f"{call['function']}({s})" for s in candidates[-2:]]
        if any(norm(quote) in norm(s) for s in candidates):
            return True
    return False
