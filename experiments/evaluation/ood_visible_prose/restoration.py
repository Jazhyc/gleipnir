"""Restore typed public text while retaining frozen tool evidence verbatim."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def digest(text: str) -> str:
    """Return a UTF-8 content identity."""
    return hashlib.sha256(text.encode()).hexdigest()


def strip_hidden(text: str) -> str:
    """Remove explicit hidden blocks, rejecting malformed or nested tags."""
    pattern = re.compile(r"</?(thinking|hidden_thinking|system_instructions)>", re.I)
    start = None
    kind = None
    output = []
    cursor = 0
    for match in pattern.finditer(text):
        closing = match[0].startswith("</")
        if not closing:
            if start is not None:
                raise ValueError("Nested hidden tags require source review")
            output.append(text[cursor : match.start()])
            start, kind = match.start(), match[1].lower()
        elif start is None or kind != match[1].lower():
            raise ValueError("Unmatched hidden tag")
        else:
            cursor, start, kind = match.end(), None, None
    if start is not None:
        raise ValueError("Unclosed hidden tag")
    output.append(text[cursor:])
    return "".join(output).strip()


def calls(value: Any) -> list[dict[str, Any]]:
    """Read tool identity from Inspect JSON or a safe literal ToolCall repr."""
    if value is None or value in ("None", "", "[]"):
        return []
    if isinstance(value, str):
        tree = ast.parse(value, mode="eval").body
        if not isinstance(tree, ast.List):
            raise ValueError("Unsupported tool-call representation")
        decoded = []
        for node in tree.elts:
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Name)
                or node.func.id != "ToolCall"
                or node.args
            ):
                raise ValueError("Unsupported tool-call syntax")
            decoded.append(
                {
                    k.arg: ast.literal_eval(k.value)
                    for k in node.keywords
                    if k.arg in {"id", "function", "arguments"}
                }
            )
        value = decoded
    if not isinstance(value, list):
        raise ValueError("Tool calls must be a list")
    return [{k: item[k] for k in ("id", "function", "arguments")} for item in value]


def typed_public(message: dict[str, Any], flattened: str) -> str:
    """Require typed boundaries and exact collation with the exported content."""
    blocks = message.get("content")
    if not isinstance(blocks, list):
        raise ValueError("Original assistant content needs typed boundaries")
    all_parts = []
    public = []
    for block in blocks:
        if block.get("type") == "text":
            text = block["text"]
            public.append(text)
        elif block.get("type") == "reasoning":
            text = block["reasoning"]
        else:
            raise ValueError("Unsupported original content type")
        all_parts.append(text)
    collations = {
        sep.join(parts) for parts in (all_parts, public) for sep in ("", "\n", "\n\n")
    }
    if flattened not in collations:
        raise ValueError("Original content does not match flattened export")
    return strip_hidden("\n".join(public))


def verify_provenance(record: dict[str, Any]) -> None:
    """Bind recovered messages to a saved, checksum-verified source snapshot."""
    p = record["provenance"]
    for key in ("source_uri", "revision", "license", "sha256"):
        if not isinstance(p.get(key), str) or not p[key]:
            raise ValueError(f"Missing provenance {key}")
    path = Path(p["source_uri"])
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != p["sha256"]:
        raise ValueError("Original snapshot checksum mismatch")
    if json.loads(payload)["messages"] != record["messages"]:
        raise ValueError("Recovered messages differ from source snapshot")


def cua_skeleton(transcript: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Separate public prose from exact role/tool evidence in a CUA transcript."""
    headers = list(re.finditer(r"^(USER|ASSISTANT|TOOL):", transcript, re.M))
    if not headers or transcript[: headers[0].start()].strip():
        raise ValueError("Unsupported CUA transcript envelope")
    skeleton = []
    prose = []
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(transcript)
        body = transcript[header.end() : end].strip()
        if header[1] == "ASSISTANT":
            marker = body.rfind("Tool calls:")
            if marker < 0:
                raise ValueError("CUA assistant without a tool anchor needs review")
            tool_text = body[marker:]
            if not re.fullmatch(r"Tool calls:\s*(?:- [\w.-]+\s*)+", tool_text):
                raise ValueError("Unsupported CUA tool anchor")
            prose.append(strip_hidden(body[:marker]))
            body = tool_text
        skeleton.append((header[1], body))
    return skeleton, prose


def restore_cua(frozen: str, public: str) -> str:
    """Restore a full public CUA export only when all non-prose evidence matches."""
    skeleton, _ = cua_skeleton(frozen)
    original, prose = cua_skeleton(public)
    if original != skeleton:
        raise ValueError("CUA original trajectory identity mismatch")
    headers = list(re.finditer(r"^ASSISTANT:", frozen, re.M))
    result = frozen
    for header, text in reversed(list(zip(headers, prose, strict=True))):
        if text:
            result = result[: header.end()] + " " + text + "\n" + result[header.end() :]
    return result


def restore(
    transcript: str, raw: list[dict[str, Any]], original: list[dict[str, Any]]
) -> str:
    """Join exact messages, inserting public text at original action positions."""
    if len(raw) != len(original):
        raise ValueError("Original message count mismatch")
    positions = list(
        re.finditer(r"^\[ASSISTANT\]\n(?=\[TOOL CALLS\]\n)", transcript, re.M)
    )
    anchors = [
        m for m in raw if m["role"] == "assistant" and calls(m.get("tool_calls"))
    ]
    if len(anchors) != len(positions):
        raise ValueError("Frozen assistant/tool anchor mismatch")
    anchor_index = 0
    pending = []
    edits = []
    for a, b in zip(raw, original, strict=True):
        if a["role"] != b["role"] or calls(a.get("tool_calls")) != calls(
            b.get("tool_calls")
        ):
            raise ValueError("Original role/tool identity mismatch")
        if a["role"] != "assistant":
            # No user, tool or system evidence is replaced by this repair.
            content = b.get("content")
            if isinstance(content, list):
                if any(part.get("type") != "text" for part in content):
                    raise ValueError("Unexpected non-assistant content type")
                alternatives = {
                    sep.join(part["text"] for part in content)
                    for sep in ("", "\n", "\n\n")
                }
            else:
                alternatives = {content}
            if a.get("content") not in alternatives:
                raise ValueError("Original non-assistant content mismatch")
            continue
        text = typed_public(b, a.get("content") or "")
        if calls(a.get("tool_calls")):
            position = positions[anchor_index]
            prefix = "".join(f"[ASSISTANT]\n{p}\n\n" for p in pending)
            replacement = prefix + "[ASSISTANT]\n" + (text + "\n" if text else "")
            edits.append((position.start(), position.end(), replacement))
            pending = []
            anchor_index += 1
        elif text:
            pending.append(text)
    result = transcript
    for start, end, replacement in reversed(edits):
        result = result[:start] + replacement + result[end:]
    if pending:
        result += "\n" + "".join(f"[ASSISTANT]\n{p}\n\n" for p in pending)
    return result
