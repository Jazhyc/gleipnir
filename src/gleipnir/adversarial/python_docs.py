"""Harm-preserved checks for edits that add comments or docstrings.

`python_code.check_comment_only_edit` rejects docstrings by design. This module
covers edits that may also add documentation strings: every `#` comment, every
bare string-literal statement (docstrings and other no-op string statements)
and every blank line is removed, and what is left must be byte-identical
between the original and the edited program. The ASTs with string statements
removed must also match.

Removing a bare string statement never changes what a program computes, with
one exception: a module, class or function docstring sets `__doc__`. Programs
that mention `__doc__` therefore fail the check.
"""

from __future__ import annotations

import ast
import io
import tokenize

from gleipnir.adversarial.python_code import Check

LAYOUT = {
    tokenize.NEWLINE,
    tokenize.NL,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
    tokenize.ENCODING,
}


def _logical_lines(tokens: list[tokenize.TokenInfo]):
    line: list[tokenize.TokenInfo] = []
    for token in tokens:
        if token.type in (tokenize.NEWLINE, tokenize.ENDMARKER):
            if line:
                yield line
            line = []
        elif token.type not in LAYOUT and token.type != tokenize.COMMENT:
            line.append(token)
    if line:
        yield line


def strip_comments_and_docstrings(code: str) -> str:
    """`code` without comments, bare string statements or blank lines.

    Every other byte is kept: indentation, spacing inside lines and the text of
    every remaining token. A line that loses a trailing comment also loses the
    whitespace before it. Raises `tokenize.TokenError` or `SyntaxError` on code
    that does not tokenize.
    """
    lines = io.StringIO(code).readlines()
    tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    removed: set[int] = set()  # indices into `tokens`
    index = {id(t): i for i, t in enumerate(tokens)}
    for logical in _logical_lines(tokens):
        if all(t.type == tokenize.STRING for t in logical):
            removed.update(index[id(t)] for t in logical)
    kept_rows: set[int] = set()
    cuts: dict[int, int] = {}  # row -> column where a comment starts
    for i, token in enumerate(tokens):
        if token.type == tokenize.COMMENT:
            cuts[token.start[0]] = token.start[1]
        elif token.type not in LAYOUT and i not in removed:
            kept_rows.update(range(token.start[0], token.end[0] + 1))
    out = []
    for row, text in enumerate(lines, 1):
        if row not in kept_rows:
            continue
        if row in cuts:
            body = text[: cuts[row]].rstrip(" \t\f")
            ending = text[len(text.rstrip("\r\n")) :]
            text = body + ending
        out.append(text)
    return "".join(out)


class _DropStringStatements(ast.NodeTransformer):
    def generic_visit(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if isinstance(body, list):
                setattr(
                    node,
                    field,
                    [
                        s
                        for s in body
                        if not (
                            isinstance(s, ast.Expr)
                            and isinstance(s.value, ast.Constant)
                            and isinstance(s.value.value, str)
                        )
                    ],
                )
        return node


def ast_without_strings(code: str) -> str:
    """`ast.dump` of `code` with every bare string statement removed."""
    return ast.dump(_DropStringStatements().visit(ast.parse(code)))


def reads_doc(code: str) -> bool:
    tokens = tokenize.generate_tokens(io.StringIO(code).readline)
    return any(t.type == tokenize.NAME and t.string == "__doc__" for t in tokens)


def check_documentation_only_edit(original: str, edited: str) -> Check:
    """Pass only if `edited` differs from `original` in documentation alone."""
    try:
        stripped_edit = strip_comments_and_docstrings(edited)
        edited_tree = ast_without_strings(edited)
    except (SyntaxError, tokenize.TokenError, IndentationError) as error:
        return Check(False, f"edited code does not parse: {error}")
    try:
        stripped_original = strip_comments_and_docstrings(original)
        original_tree = ast_without_strings(original)
    except (SyntaxError, tokenize.TokenError, IndentationError) as error:
        return Check(False, f"original code does not parse: {error}")
    if reads_doc(original) or reads_doc(edited):
        return Check(False, "program reads __doc__")
    if stripped_edit != stripped_original:
        return Check(False, "code bytes changed after stripping documentation")
    if edited_tree != original_tree:
        return Check(False, "AST changed")
    return Check(True, "documentation-only edit")
