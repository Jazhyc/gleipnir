"""Mechanical confession strip (stdlib plus the repo's Python-doc helpers).

`strip_messages` returns a copy of a cut run with:
- every assistant message's own text (prose outside tool calls) emptied;
- `#` comments and bare string statements (docstrings and other no-op string
  statements) removed from every piece of Python source in tool-call
  arguments: the python tool's code, text_editor `file_text`/`new_str`/
  `old_str` on `.py` paths, bash heredocs written to a `.py` file or fed to a
  Python interpreter, and `python -c` strings.

Everything else stays byte-identical: tool outputs, user turns, shell text,
printed strings, identifiers. A piece is stripped with
`python_docs.strip_comments_and_docstrings` (which also drops blank lines) and
kept only if `check_documentation_only_edit` passes on the original and the
stripped piece (dedented first when a fragment is indented). A piece that does
not parse, or whose stripped form fails the check, is left unchanged and
counted.
"""

from __future__ import annotations

import ast
import copy
import io
import textwrap
import tokenize
import warnings
from collections import Counter
from typing import Any

from experiments.adversarial_natural_cheats.trajectories import (
    AFTER_QUOTE,
    PYTHON_C,
    bash_chunks,
)
from gleipnir.adversarial.python_docs import (
    _logical_lines,
    check_documentation_only_edit,
    strip_comments_and_docstrings,
)

PYTHON_TOOL_KEYS = ("code", "command")
EDITOR_KEYS = ("file_text", "new_str", "old_str", "content")


def parseable(code: str) -> str | None:
    """`code`, dedented if needed, when it parses; else None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        for candidate in (code, textwrap.dedent(code)):
            try:
                ast.parse(candidate)
                return candidate
            except (SyntaxError, ValueError):
                continue
    return None


def doc_counts(code: str) -> tuple[int, int]:
    """(# comments, bare string statements) in code that tokenizes."""
    tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    comments = sum(t.type == tokenize.COMMENT for t in tokens)
    strings = sum(
        all(t.type == tokenize.STRING for t in line) for line in _logical_lines(tokens)
    )
    return comments, strings


def strip_piece(code: str, kind: str, stats: Counter[str]) -> str:
    """One piece of Python source without comments or docstrings, or unchanged."""
    stats["pieces"] += 1
    stats[f"{kind}:pieces"] += 1
    original = parseable(code)
    if original is None:
        stats["parse_failed"] += 1
        stats[f"{kind}:parse_failed"] += 1
        return code
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            stripped = strip_comments_and_docstrings(code)
            comments, strings = doc_counts(code)
        except (SyntaxError, tokenize.TokenError, IndentationError):
            stats["parse_failed"] += 1
            stats[f"{kind}:parse_failed"] += 1
            return code
        if comments == 0 and strings == 0:
            return code  # only blank lines would go: leave the piece as it was
        target = parseable(stripped)
        ok = target is not None and check_documentation_only_edit(original, target).ok
    if not ok:
        stats["check_failed"] += 1
        stats[f"{kind}:check_failed"] += 1
        return code
    stats["pieces_changed"] += 1
    stats["comments_removed"] += comments
    stats["docstrings_removed"] += strings
    return stripped


def strip_python_c(text: str, stats: Counter[str]) -> str:
    """`python -c` strings stripped where bash passes the quoted text literally
    (natural-cheats' rule); anything else is left unchanged and counted."""
    out, pos = [], 0
    while match := PYTHON_C.search(text, pos):
        quote, start = match.group(1), match.end()
        close = text.find(quote, start)
        code = text[start:close] if close >= 0 else ""
        if close < 0 or not AFTER_QUOTE.match(text, close + 1):
            stats["python_c:quoting_skipped"] += 1
            break
        if quote == '"' and any(c in code for c in "\\$`"):
            stats["python_c:escapes_skipped"] += 1
            break
        out.append(text[pos:start])
        out.append(strip_piece(code, "python_c", stats))
        pos = close
    out.append(text[pos:])
    return "".join(out)


def strip_bash(command: str, stats: Counter[str]) -> str:
    out = []
    for kind, text in bash_chunks(command, stats):
        if kind == "shell":
            out.append(strip_python_c(text, stats))
        elif kind in ("py_file", "python"):
            out.append(strip_piece(text, f"heredoc_{kind}", stats))
        else:
            out.append(text)
    return "".join(out)


def strip_call(call: dict[str, Any], stats: Counter[str]) -> dict[str, Any]:
    new = copy.deepcopy(call)
    args = new.get("arguments") or {}
    name = new["function"]
    if name == "python":
        for key in PYTHON_TOOL_KEYS:
            if isinstance(args.get(key), str):
                args[key] = strip_piece(args[key], "python_tool", stats)
    elif name == "text_editor":
        is_py = str(args.get("path", "")).endswith(".py")
        for key in EDITOR_KEYS:
            if isinstance(args.get(key), str):
                if is_py:
                    args[key] = strip_piece(args[key], f"editor_{key}", stats)
                else:
                    stats["editor:non_py_path_left"] += 1
    elif name == "bash" and isinstance(args.get("command"), str):
        args["command"] = strip_bash(args["command"], stats)
    else:
        stats[f"call_left:{name}"] += 1
    return new


def strip_messages(
    messages: list[dict[str, Any]], stats: Counter[str]
) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        if m["role"] == "assistant":
            if m.get("text"):
                stats["prose_messages_removed"] += 1
                stats["prose_chars_removed"] += len(m["text"])
            m = {
                **m,
                "text": "",
                "tool_calls": [strip_call(c, stats) for c in m.get("tool_calls") or []],
            }
        out.append(m)
    return out


TOTALS = (
    "prose_messages_removed",
    "prose_chars_removed",
    "pieces",
    "pieces_changed",
    "comments_removed",
    "docstrings_removed",
    "parse_failed",
    "check_failed",
)


def summary(stats: Counter[str]) -> dict[str, int]:
    """The per-row record: headline counts plus every detailed counter."""
    return {
        **{k: stats.get(k, 0) for k in TOTALS},
        "detail": {k: v for k, v in sorted(stats.items()) if k not in TOTALS},
    }
