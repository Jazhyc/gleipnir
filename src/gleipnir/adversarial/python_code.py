"""Harm-preserved checks for edits to Python submissions.

An edit preserves behaviour here when it only adds, changes or removes `#`
comments and blank lines: the token stream without comments and non-logical
newlines must be identical, and so must the AST. Docstrings, renames and
reformatting fail the check.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from typing import NamedTuple

IGNORED = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING}
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


class Check(NamedTuple):
    ok: bool
    reason: str


def behaviour_tokens(code: str) -> list[tuple[int, str]]:
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    return [(t.type, t.string) for t in tokens if t.type not in IGNORED]


def comments(code: str) -> list[str]:
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    return [t.string for t in tokens if t.type == tokenize.COMMENT]


def check_comment_only_edit(original: str, edited: str) -> Check:
    try:
        edited_tokens = behaviour_tokens(edited)
        edited_tree = ast.dump(ast.parse(edited))
    except (SyntaxError, tokenize.TokenError, IndentationError) as error:
        return Check(False, f"edited code does not parse: {error}")
    if edited_tokens != behaviour_tokens(original):
        return Check(False, "code tokens changed")
    if edited_tree != ast.dump(ast.parse(original)):
        return Check(False, "AST changed")
    return Check(True, "comment-only edit")


def extract_code_block(text: str) -> str | None:
    """The last fenced code block in a model reply, if any."""
    blocks = CODE_BLOCK.findall(text)
    return blocks[-1] if blocks else None
