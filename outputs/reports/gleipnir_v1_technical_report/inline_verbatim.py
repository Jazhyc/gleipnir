"""Inline \\VerbatimInput files into a .tex file for the arXiv bundle.

arXiv's build did not find the prompt files read with \\VerbatimInput, so the
arXiv copy of the appendix carries their text in Verbatim environments
instead. The repository sources keep \\VerbatimInput.

    python inline_verbatim.py arxiv/sections/appendix.tex
"""

import re
import sys
from pathlib import Path

PATTERN = re.compile(r"\\VerbatimInput(\[[^\]]*\])?\{([^}]+)\}")


def inline(tex_path: Path, root: Path) -> int:
    text = tex_path.read_text(encoding="utf-8")
    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        options = match.group(1) or ""
        body = (root / match.group(2)).read_text(encoding="utf-8").rstrip("\n")
        if "\\end{Verbatim}" in body:
            raise ValueError(f"{match.group(2)} contains \\end{{Verbatim}}")
        count += 1
        return f"\\begin{{Verbatim}}{options}\n{body}\n\\end{{Verbatim}}"

    tex_path.write_text(PATTERN.sub(replace, text), encoding="utf-8")
    return count


if __name__ == "__main__":
    path = Path(sys.argv[1])
    print(f"inlined {inline(path, path.parent.parent)} verbatim files into {path}")
