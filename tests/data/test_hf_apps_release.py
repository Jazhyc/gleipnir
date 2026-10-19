from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import release_hf_apps_dataset as release


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write tiny upstream and control files and pin the script to their hashes."""
    upstream, controls = tmp_path / "upstream", tmp_path / "controls"
    contents = {}
    rows = {}
    for relative in release.UPSTREAM_FILES | release.CONTROL_FILES:
        n = 2 if relative.endswith(".jsonl") else 0
        contents[relative] = "".join(f'{{"id": "{relative}-{i}"}}\n' for i in range(n))
        if relative in release.EXPECTED_ROWS:
            rows[relative] = n
    hashes = {k: digest(v.encode()) for k, v in contents.items()}
    for variant in ("neutral", "injection_aware"):
        rows_path = f"data/apps/honest_controls/eval_{variant}.jsonl"
        contents[rows_path.replace(".jsonl", ".manifest.json")] = json.dumps(
            {
                "output": {"sha256": hashes[rows_path]},
                "source": {
                    "hf_revision": release.UPSTREAM_REVISION,
                    "sha256": hashes[f"data/apps/prompts/eval_{variant}.jsonl"],
                },
            }
        )
    for relative, text in contents.items():
        root = controls if relative in release.CONTROL_FILES else upstream
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)
    pinned = lambda files: {  # noqa: E731
        k: hashes[k] if v is not None else None for k, v in files.items()
    }
    monkeypatch.setattr(release, "UPSTREAM_FILES", pinned(release.UPSTREAM_FILES))
    monkeypatch.setattr(release, "CONTROL_FILES", pinned(release.CONTROL_FILES))
    monkeypatch.setattr(release, "EXPECTED_ROWS", rows)
    monkeypatch.setattr(
        release,
        "hf_hub_download",
        lambda repo, relative, repo_type, revision: str(upstream / relative),
    )
    return controls


def test_stage_writes_verified_manifest(tmp_path: Path, sources: Path) -> None:
    staged = release.stage(tmp_path / "out", sources)

    manifest = release.load_manifest(staged)
    assert manifest["repo_name"] == release.REPO_NAME
    assert manifest["upstream_revision"] == release.UPSTREAM_REVISION
    assert "README.md" in manifest["files"]
    assert set(release.CONTROL_FILES) <= set(manifest["files"])
    with pytest.raises(FileExistsError):
        release.stage(tmp_path / "out", sources)


def test_stage_rejects_control_drift(tmp_path: Path, sources: Path) -> None:
    path = sources / "data/apps/honest_controls/eval_neutral.jsonl"
    path.write_text(path.read_text() + "{}\n")

    with pytest.raises(ValueError, match="frozen checksum drift"):
        release.stage(tmp_path / "out", sources)
    assert not list((tmp_path / "out").iterdir())


def test_upload_defaults_to_private(
    tmp_path: Path, sources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = release.stage(tmp_path / "out", sources)
    calls = []

    class FakeApi:
        def create_repo(self, **kwargs):
            calls.append(("create", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs))

    monkeypatch.setattr(release, "HfApi", FakeApi)

    repo_id = release.upload(staged, "Jazhyc", public=False, allow_existing=False)

    assert repo_id == f"Jazhyc/{release.REPO_NAME}"
    assert calls[0] == (
        "create",
        {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "private": True,
            "exist_ok": False,
        },
    )
    assert calls[1][1]["repo_type"] == "dataset"
