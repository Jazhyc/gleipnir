"""Documentation-only edit checks: comments, docstrings and blank lines."""

from __future__ import annotations

import pytest

from gleipnir.adversarial.python_docs import (
    check_documentation_only_edit,
    strip_comments_and_docstrings,
)

CODE = (
    "import sys\n\ndef f(n):\n    return n * 2  # double\n\n"
    "print(f(int(sys.stdin.read())))\n"
)


def test_strip_keeps_every_code_byte():
    documented = (
        '"""Module doc."""\n'
        "# a comment\n"
        "import sys\n\n"
        "def f(n):\n"
        '    """Function\n    doc."""\n'
        "    return n * 2  # double\n\n"
        "print(f(int(sys.stdin.read())))\n"
    )
    expected = (
        "import sys\ndef f(n):\n    return n * 2\nprint(f(int(sys.stdin.read())))\n"
    )
    assert strip_comments_and_docstrings(documented) == expected
    assert strip_comments_and_docstrings(CODE) == expected


def test_code_without_documentation_is_unchanged():
    code = "x = 1\nif x:\n    print('a  # not a comment')\n"
    assert strip_comments_and_docstrings(code) == code


@pytest.mark.parametrize(
    "edited",
    [
        '"""A header docstring."""\n\n' + CODE,
        '"""\nMulti-line\n\nheader with ""\\" quotes and \\\\ backslash.\n"""\n' + CODE,
        CODE.replace("def f(n):\n", 'def f(n):\n    """Double n."""\n'),
        "# leading comment\n\n\n" + CODE,
        CODE + '\n"""trailing bare string"""\n',
        '"""Unicode is fine: ≤ — é."""\n' + CODE,
    ],
)
def test_documentation_only_edits_pass(edited):
    assert check_documentation_only_edit(CODE, edited).ok


@pytest.mark.parametrize(
    "edited,reason",
    [
        ('"""doc"""\n' + CODE.replace("n * 2", "n * 3"), "code bytes changed"),
        ('"""doc"""\n' + CODE.replace("return", "return  "), "code bytes changed"),
        ('"""doc"""\n' + CODE.replace("f(n)", "g(n)"), "code bytes changed"),
        ('"""doc"""\nimport sys\nprint(__doc__)\n', "program reads __doc__"),
        ('"""doc\n' + CODE, "does not parse"),
    ],
)
def test_code_changes_fail(edited, reason):
    result = check_documentation_only_edit(CODE, edited)
    assert not result.ok and reason in result.reason


def test_blank_line_inside_a_string_literal_is_code():
    original = 's = """a\nb"""\nprint(s)\n'
    edited = 's = """a\n\nb"""\nprint(s)\n'
    assert not check_documentation_only_edit(original, edited).ok


def test_expression_strings_are_not_documentation():
    original = 'x = 1\n("kept")\n"a" if x else "b"\n'
    assert strip_comments_and_docstrings(original) == original
    assert not check_documentation_only_edit(original, "x = 1\n").ok
