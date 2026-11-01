"""Inspect agent runs as action-only monitor trajectories (stdlib only).

Inspect messages are first normalized to plain dicts (`normalize`), so the
log-reading step and the row-building step can run in different environments.
`render` produces the strict-OOD block format of
`gleipnir.adversarial.multiagent.render_history`:

    [USER]                  user text (task prompt, scaffold "failed" replies)
    [ASSISTANT]
    [TOOL CALLS]            name({json arguments}), one call per line
    [TOOL]                  tool output as the agent saw it (errors as `Error: ...`)

joined by blank lines. The system prompt and all assistant prose are dropped;
an assistant turn without tool calls leaves no block. Tool outputs are shown
raw (Inspect stores plain text, not JSON).

`strip_call` removes Python `#` comments from code inside tool-call
arguments with `tokenize`; code that does not tokenize is left unchanged and
counted. Docstrings and every code token stay as they were.
"""

from __future__ import annotations

import ast
import copy
import io
import json
import re
import tokenize
import warnings
from collections import Counter
from typing import Any

from gleipnir.adversarial.python_code import behaviour_tokens

FAILURE_REPLY = "Your implementation failed the tests"


# ---------------------------------------------------------------- messages


def normalize(message: Any) -> dict[str, Any]:
    """An Inspect ChatMessage as a plain, JSON-serializable dict."""
    role = message.role
    out: dict[str, Any] = {"role": role, "text": message.text or ""}
    if role == "assistant":
        out["tool_calls"] = [
            {"function": call.function, "arguments": call.arguments or {}}
            for call in getattr(message, "tool_calls", None) or []
        ]
    elif role == "tool":
        error = getattr(message, "error", None)
        out["function"] = getattr(message, "function", None)
        out["error"] = (
            None if error is None else {"type": error.type, "message": error.message}
        )
    return out


def call_text(call: dict[str, Any]) -> str:
    arguments = json.dumps(call.get("arguments") or {}, ensure_ascii=False)
    return f"{call['function']}({arguments})"


def render(messages: list[dict[str, Any]]) -> str:
    blocks = []
    for message in messages:
        role = message["role"]
        if role == "user":
            blocks.append(f"[USER]\n{message['text']}")
        elif role == "assistant":
            calls = [call_text(call) for call in message.get("tool_calls") or []]
            if calls:
                blocks.append("[ASSISTANT]\n[TOOL CALLS]\n" + "\n".join(calls))
        elif role == "tool":
            # What the agent saw: Inspect's OpenAI request sends an errored tool
            # result as "Error: <message>" in place of its (usually empty) text.
            error = message.get("error")
            text = f"Error: {error['message']}" if error else message["text"]
            blocks.append(f"[TOOL]\n{text}")
    return "\n\n".join(blocks)


def tool_call_turns(messages: list[dict[str, Any]]) -> list[int]:
    return [
        i
        for i, m in enumerate(messages)
        if m["role"] == "assistant" and m.get("tool_calls")
    ]


def failure_replies(messages: list[dict[str, Any]]) -> list[int]:
    """Indices of the scaffold's "Your implementation failed the tests" replies."""
    return [
        i
        for i, m in enumerate(messages)
        if m["role"] == "user" and m["text"].startswith(FAILURE_REPLY)
    ]


def nearest_turn(turns: list[int], length: int) -> int:
    """The tool-call turn nearest `length`; ties go to the shorter cut.

    A length past the run's last turn snaps to that turn (clipped to the run).
    """
    if not turns:
        raise ValueError("run has no assistant tool-call turn")
    return min(turns, key=lambda j: (abs(j - length), j))


def first_turn_after(messages: list[dict[str, Any]], index: int) -> int | None:
    return next((j for j in tool_call_turns(messages) if j > index), None)


def cut(messages: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    """Messages up to and including `index`, so a cut after an assistant turn
    never contains that turn's tool results."""
    return messages[: index + 1]


# ---------------------------------------------------------------- comments


def strip_comments(code: str) -> tuple[str, int] | None:
    """`code` without `#` comments, and how many were removed.

    Returns None if the code does not tokenize. A comment-only line is
    removed with its newline; an inline comment is removed with the
    whitespace before it. Every other byte stays.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, SyntaxError):
        return None
    by_row = {t.start[0]: t.string for t in tokens if t.type == tokenize.COMMENT}
    if not by_row:
        return code, 0
    out = []
    # StringIO.readlines splits exactly as tokenize's readline did.
    for row, line in enumerate(io.StringIO(code).readlines(), 1):
        comment = by_row.get(row)
        if comment is None:
            out.append(line)
            continue
        body = line.rstrip("\r\n")
        if not body.endswith(comment):
            return None
        kept = body[: len(body) - len(comment)].rstrip()
        if kept:
            out.append(kept + line[len(body) :])
    return "".join(out), len(by_row)


def comment_only_change(original: str, stripped: str) -> bool:
    """Same code tokens (and AST, where the original parses) without comments."""
    try:
        if behaviour_tokens(original) != behaviour_tokens(stripped):
            return False
    except (tokenize.TokenError, SyntaxError):
        return False
    with warnings.catch_warnings():  # agents' code has invalid escapes
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            tree = ast.dump(ast.parse(original))
        except SyntaxError:
            return True  # fragments (e.g. an indented str_replace) have no AST
        try:
            return ast.dump(ast.parse(stripped)) == tree
        except SyntaxError:
            return False


def strip_piece(code: str, kind: str, stats: Counter[str]) -> str:
    """Strip one piece of code, counting the outcome under `kind`."""
    stats[f"{kind}:pieces"] += 1
    result = strip_comments(code)
    if result is None:
        stats[f"{kind}:tokenize_failed"] += 1
        stats["tokenize_failed"] += 1
        return code
    stripped, removed = result
    if removed and not comment_only_change(code, stripped):
        stats[f"{kind}:check_failed"] += 1
        stats["check_failed"] += 1
        return code
    if removed:
        stats[f"{kind}:changed"] += 1
        stats["comments_removed"] += removed
    return stripped


HEREDOC = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
PY_FILE = re.compile(
    r"(?:>>?|\btee\b(?:\s+-a)?)\s*['\"]?[^\s'\";|&<>]*\.py(?=$|[\s'\";|&)<>])"
)
PY_INTERPRETER = re.compile(r"\bpython(?:3(?:\.\d+)?)?(?:\s+-[A-Za-z]+)*(?:\s+-)?\s*$")
PYTHON_C = re.compile(r"\bpython(?:3(?:\.\d+)?)?(?:\s+-[A-Za-z]+)*\s+-c\s+(['\"])")
AFTER_QUOTE = re.compile(r"$|[\s;|&)]")
EDITOR_COMMANDS = {"view", "create", "str_replace", "insert", "undo_edit"}


def heredoc_target(line: str, match: re.Match) -> str | None:
    """'py_file' for a heredoc written to a .py file, 'python' for one fed to a
    Python interpreter, else None (a heredoc of other text)."""
    if PY_FILE.search(line):
        return "py_file"
    if PY_INTERPRETER.search(line[: match.start()]):
        return "python"
    return None


def closes_heredoc(line: str, delimiter: str, tabs: bool) -> bool:
    text = line.rstrip("\r\n")
    return (text.lstrip("\t") if tabs else text) == delimiter


def bash_chunks(command: str, stats: Counter[str]) -> list[tuple[str | None, str]]:
    """Split a bash command into shell text and heredoc bodies.

    Returns (kind, text) pairs: kind "shell" for command text, or the
    heredoc's `heredoc_target` (None for a non-Python heredoc).
    """
    lines = command.splitlines(keepends=True)
    chunks: list[tuple[str | None, str]] = []
    shell: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        shell.append(line)
        match = HEREDOC.search(line)
        i += 1
        if not match:
            continue
        if len(HEREDOC.findall(line)) > 1:
            stats["heredoc:multiple_on_one_line"] += 1
        tabs, delimiter = match.group(1) == "-", match.group(3)
        end = next(
            (
                j
                for j in range(i, len(lines))
                if closes_heredoc(lines[j], delimiter, tabs)
            ),
            None,
        )
        if end is None:
            stats["heredoc:unterminated"] += 1
            shell.extend(lines[i:])
            break
        chunks.append(("shell", "".join(shell)))
        shell = [lines[end]]
        chunks.append((heredoc_target(line, match), "".join(lines[i:end])))
        i = end + 1
    chunks.append(("shell", "".join(shell)))
    return chunks


def strip_python_c(text: str, stats: Counter[str]) -> str:
    """Strip `python -c '...'` arguments whose quoted text is the code itself.

    Single-quoted text is literal; double-quoted text is used only without a
    backslash, `$` or backtick, so bash changes nothing. The closing quote must
    end the word (this rules out quote-splicing such as `'...'"'"'...'`).
    Anything else is left unchanged, with the rest of the text, and counted.
    """
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


def strip_bash(
    command: str, stats: Counter[str], python_heredocs: bool, python_c: bool
) -> str:
    out = []
    for kind, text in bash_chunks(command, stats):
        if kind == "shell":
            out.append(strip_python_c(text, stats) if python_c else text)
        elif kind == "py_file" or (kind == "python" and python_heredocs):
            out.append(strip_piece(text, f"heredoc_{kind}", stats))
        else:
            out.append(text)
    return "".join(out)


def strip_call(
    call: dict[str, Any],
    stats: Counter[str],
    *,
    python_heredocs: bool = True,
    python_c: bool = True,
) -> dict[str, Any]:
    """A copy of one tool call with Python comments removed from its code.

    Covers the python tool's `code`; text_editor `file_text`/`new_str` on
    `.py` paths; and bash heredocs written to `.py` files, plus (if enabled)
    heredocs fed to a Python interpreter and `python -c` strings. Any other
    argument is left unchanged.
    """
    new = copy.deepcopy(call)
    arguments = new.get("arguments") or {}
    name = new["function"]
    if name == "python" and isinstance(arguments.get("code"), str):
        arguments["code"] = strip_piece(arguments["code"], "python_tool", stats)
    elif name == "text_editor":
        path = str(arguments.get("path", ""))
        if arguments.get("command") not in EDITOR_COMMANDS:
            stats["uncovered_call:text_editor_malformed"] += 1
        for key in ("file_text", "new_str"):
            if isinstance(arguments.get(key), str):
                if path.endswith(".py"):
                    arguments[key] = strip_piece(arguments[key], "text_editor", stats)
                else:
                    stats["text_editor:non_py_path_skipped"] += 1
    elif name == "bash" and isinstance(arguments.get("command"), str):
        arguments["command"] = strip_bash(
            arguments["command"], stats, python_heredocs, python_c
        )
    elif name != "submit":
        stats[f"uncovered_call:{name}"] += 1
    return new


def strip_messages(
    messages: list[dict[str, Any]], stats: Counter[str], **options: bool
) -> list[dict[str, Any]]:
    """Messages with every assistant tool call stripped; outputs unchanged."""
    out = []
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            message = {
                **message,
                "tool_calls": [
                    strip_call(c, stats, **options) for c in message["tool_calls"]
                ],
            }
        out.append(message)
    return out
