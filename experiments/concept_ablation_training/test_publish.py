"""Publication identity guards must reject drift before any remote mutation."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments.concept_ablation_training import publish


@pytest.mark.parametrize(
    ("namespace", "title", "private", "error"),
    [
        ("other", "Gleipnir Prompt Injection Monitoring", False, "account"),
        ("Jazhyc", "Other collection", False, "collection"),
        ("Jazhyc", "Gleipnir Prompt Injection Monitoring", True, "visibility"),
    ],
)
def test_wrong_destination_never_mutates(
    monkeypatch, tmp_path, namespace, title, private, error
):
    api = Mock()
    api.whoami.return_value = {"name": namespace}
    api.get_collection.return_value = SimpleNamespace(title=title, private=private)
    monkeypatch.setattr(publish, "dotenv_values", lambda _: {"HF_TOKEN": "mock"})
    monkeypatch.setattr(publish, "HfApi", lambda **_: api)
    with pytest.raises(ValueError, match=error):
        publish.publish(tmp_path)
    api.create_repo.assert_not_called()
    api.upload_folder.assert_not_called()
    api.add_collection_item.assert_not_called()


def test_failed_parity_blocks_staging(monkeypatch, tmp_path):
    monkeypatch.setattr(publish, "CAMPAIGN", tmp_path)
    monkeypatch.setattr(publish, "OUTPUT", tmp_path)

    def read(path):
        if path.name == "runner.json":
            return {"status": "complete"}
        return {"passed": False}

    monkeypatch.setattr(publish, "read", read)
    with pytest.raises(ValueError, match="successful parity"):
        publish.stage()
    assert not (tmp_path / "release").exists()
