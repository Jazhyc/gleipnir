"""Exercise public upload routing and verification without calling paid services."""

import json
import sys
from types import SimpleNamespace

import pytest

from experiments.student_injection_awareness import publish


@pytest.mark.parametrize("corrupt", [False, True])
def test_upload_requires_remote_adapter_checksums(tmp_path, monkeypatch, corrupt):
    calls = []
    output = tmp_path / "results"
    monkeypatch.setattr(publish, "OUTPUT", output)
    monkeypatch.setattr(publish, "dotenv_values", lambda _: {"HF_TOKEN": "mock-token"})
    monkeypatch.setattr(sys, "argv", ["publish", "--size", "4b"])

    def stage(size, variant):
        release = tmp_path / f"{size}-{variant}"
        release.mkdir()
        manifest = {
            "files_sha256": {
                "adapter_model.safetensors": "master",
                "vllm/adapter_model.safetensors": "serving",
            }
        }
        (release / "release_manifest.json").write_text(json.dumps(manifest))
        return release

    monkeypatch.setattr(publish, "stage", stage)

    class FakeApi:
        def __init__(self, token):
            assert token == "mock-token"

        def whoami(self):
            return {"name": "test-namespace"}

        def create_repo(self, **kwargs):
            calls.append(kwargs)

        def upload_folder(self, **kwargs):
            return SimpleNamespace(oid="revision")

        def model_info(self, repo_id, **kwargs):
            assert kwargs == {"revision": "revision", "files_metadata": True}
            return SimpleNamespace(
                sha="revision",
                siblings=[
                    SimpleNamespace(rfilename="release_manifest.json", lfs=None),
                    SimpleNamespace(
                        rfilename="adapter_model.safetensors",
                        lfs=SimpleNamespace(sha256="wrong" if corrupt else "master"),
                    ),
                    SimpleNamespace(
                        rfilename="vllm/adapter_model.safetensors",
                        lfs=SimpleNamespace(sha256="serving"),
                    ),
                ],
            )

    monkeypatch.setattr(publish, "HfApi", FakeApi)
    if corrupt:
        with pytest.raises(ValueError, match="uploaded adapter checksum mismatch"):
            publish.main()
        assert not (output / "4b/regular/upload.json").exists()
    else:
        publish.main()
        assert len(calls) == 2
        assert all(c["repo_id"].startswith("test-namespace/") for c in calls)
        assert all(c["private"] is False and c["exist_ok"] is False for c in calls)
        for variant in publish.VARIANTS:
            receipt = json.loads((output / "4b" / variant / "upload.json").read_text())
            assert receipt["remote_verified"] is True


def test_stage_rejects_failed_parity(tmp_path, monkeypatch):
    directory = tmp_path / "4b/regular"
    directory.mkdir(parents=True)
    (directory / "serving_parity.json").write_text('{"passed":false}')
    monkeypatch.setattr(publish, "OUTPUT", tmp_path)
    with pytest.raises(ValueError, match="release serving parity did not pass"):
        publish.stage("4b", "regular")
