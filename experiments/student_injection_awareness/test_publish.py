"""Exercise public upload routing and verification without calling paid services."""

import json
import sys
from types import SimpleNamespace

import pytest

from experiments.student_injection_awareness import publish


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("selected", [None, "regular", "injection_aware"])
def test_upload_requires_remote_adapter_checksums(
    tmp_path, monkeypatch, corrupt, selected
):
    calls = []
    output = tmp_path / "results"
    monkeypatch.setattr(publish, "OUTPUT", output)
    monkeypatch.setattr(publish, "dotenv_values", lambda _: {"HF_TOKEN": "mock-token"})
    argv = ["publish", "--size", "4b"]
    if selected:
        argv.extend(["--variant", selected])
    variants = (selected,) if selected else publish.VARIANTS
    monkeypatch.setattr(sys, "argv", argv)

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
            assert kwargs["allow_patterns"] == list(publish.PUBLIC_FILES)
            assert not any(
                "metadata" in name or "provenance" in name or "evaluation" in name
                for name in kwargs["allow_patterns"]
            )
            return SimpleNamespace(oid="revision")

        def model_info(self, repo_id, **kwargs):
            assert kwargs == {"revision": "revision", "files_metadata": True}
            return SimpleNamespace(
                sha="revision",
                siblings=[
                    *[
                        SimpleNamespace(rfilename=name, lfs=None)
                        for name in publish.PUBLIC_FILES
                        if not name.endswith(".safetensors")
                    ],
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
        assert not (output / "4b" / variants[0] / "upload.json").exists()
    else:
        publish.main()
        assert len(calls) == len(variants)
        assert [c["repo_id"] for c in calls] == [
            f"test-namespace/4b-{variant}" for variant in variants
        ]
        assert all(c["repo_id"].startswith("test-namespace/") for c in calls)
        assert all(c["private"] is False and c["exist_ok"] is False for c in calls)
        for variant in variants:
            receipt = json.loads((output / "4b" / variant / "upload.json").read_text())
            assert receipt["remote_verified"] is True


def test_stage_rejects_failed_parity(tmp_path, monkeypatch):
    directory = tmp_path / "4b/regular"
    directory.mkdir(parents=True)
    (directory / "serving_parity.json").write_text('{"passed":false}')
    monkeypatch.setattr(publish, "OUTPUT", tmp_path)
    with pytest.raises(ValueError, match="release serving parity did not pass"):
        publish.stage("4b", "regular")
