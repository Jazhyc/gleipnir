"""Documentation layout: complete indexes, dated headers and live links."""

import os
import re
import subprocess

import pytest

from tests.helpers.paths import ROOT

DOCS = ROOT / "docs"
LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _records():
    for folder in ("findings", "decisions", "plans"):
        for path in sorted((DOCS / folder).rglob("*.md")):
            if path.name != "README.md":
                yield path


def test_every_record_is_indexed():
    index = (DOCS / "README.md").read_text()
    decisions = (DOCS / "decisions/README.md").read_text()
    for path in _records():
        rel = path.relative_to(DOCS).as_posix()
        assert f"]({rel})" in index, rel
        if rel.startswith("decisions/"):
            assert f"]({path.relative_to(DOCS / 'decisions').as_posix()})" in decisions


def test_records_sit_in_category_folders():
    for path in _records():
        parts = path.relative_to(DOCS).parts
        expected = 2 if parts[0] == "plans" else 3
        assert len(parts) == expected, path


def test_every_record_has_status_and_date():
    date = re.compile(r"\b20\d\d-\d\d-\d\d\b|\b\d{1,2} [A-Z][a-z]+ 20\d\d\b")
    for path in _records():
        head = "\n".join(path.read_text().split("\n")[:16])
        assert "Status" in head and re.search(r"Status\**:", head), path
        assert date.search(head), path


def test_decision_statuses_use_the_vocabulary_and_link_to_current():
    pattern = re.compile(
        r"^Status: (current\.|historical; governs no future work\.|"
        r"superseded by \[[^\]]+\]\(([^)]+)\).*)$",
        re.M,
    )
    current = (DOCS / "current.md").read_text()
    for path in (DOCS / "decisions").rglob("*.md"):
        if path.name == "README.md":
            continue
        head = "\n".join(path.read_text().split("\n")[:16])
        match = pattern.search(head)
        assert match, path
        if match.group(2):
            target = (path.parent / match.group(2)).resolve()
            assert target.read_text().count("Status: current.") == 1, (path, target)
        if match.group(1) == "current.":
            rel = path.relative_to(DOCS).as_posix()
            assert f"]({rel}" in current, (
                f"current decision missing from current.md: {rel}"
            )


def _relative_links(text):
    for target in LINK.findall(text):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith(
            ("#", "/", "<")
        ):
            continue
        path = target.split("#")[0]
        if path:
            yield target, path


@pytest.mark.parametrize("scope", ["docs", "experiments", "other"])
def test_markdown_links_resolve(scope):
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    ).stdout.split()
    broken = []
    for rel in tracked:
        top = rel.split("/")[0]
        if (scope == "other") == (top in ("docs", "experiments")) or (
            scope != "other" and top != scope
        ):
            continue
        source = ROOT / rel
        if not source.is_file():
            continue
        for target, path in _relative_links(source.read_text(errors="ignore")):
            resolved = os.path.normpath(source.parent / path)
            if os.path.relpath(resolved, ROOT).split(os.sep)[0] in (
                "results",
                "data",
                "logs",
            ):
                continue
            if not os.path.exists(resolved):
                broken.append(f"{rel}: {target}")
    assert not broken, "\n".join(broken)
