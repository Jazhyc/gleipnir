from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import build_eval_datasets as builder


def test_upstream_files_cover_pinned_sources() -> None:
    id_files = builder.upstream_files("id")
    ood_files = builder.upstream_files("ood")

    assert [path.name for *_, path in id_files] == [
        "stride_test.parquet",
        "gloom_exfiltration.parquet",
    ]
    assert len(ood_files) == len(builder.ood_source.OOD_SOURCES) + 1
    assert all(expected for _, _, _, expected, _ in id_files + ood_files)
    assert {revision for _, revision, *_ in ood_files} == {
        builder.ood_source.SOURCE_REVISION
    }


def test_fetch_downloads_once_and_rejects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream = tmp_path / "upstream.parquet"
    upstream.write_bytes(b"rows")
    local = tmp_path / "local/source.parquet"
    expected = hashlib.sha256(b"rows").hexdigest()
    downloads = []

    def download(repo, repo_file, repo_type, revision):
        downloads.append((repo, repo_file, repo_type, revision))
        return str(upstream)

    monkeypatch.setattr(builder, "hf_hub_download", download)
    monkeypatch.setattr(
        builder,
        "upstream_files",
        lambda suite: [("org/data", "rev", "data/a.parquet", expected, local)],
    )

    builder.fetch("id")
    builder.fetch("id")
    assert local.read_bytes() == b"rows"
    assert downloads == [("org/data", "data/a.parquet", "dataset", "rev")]

    local.write_bytes(b"edited")
    with pytest.raises(ValueError, match="upstream checksum drift"):
        builder.fetch("id")


def test_verify_reports_missing_and_changed_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good, changed = tmp_path / "good.jsonl", tmp_path / "changed.jsonl"
    good.write_text("a\n")
    changed.write_text("b\n")
    good_hash = hashlib.sha256(b"a\n").hexdigest()
    monkeypatch.setattr(
        builder,
        "EXPECTED",
        {
            "id": {good: good_hash},
            "ood": {good: good_hash, changed: good_hash, tmp_path / "x": good_hash},
        },
    )

    assert builder.verify("id") == {str(good): good_hash}
    with pytest.raises(ValueError, match="changed.jsonl.*missing"):
        builder.verify("ood")
